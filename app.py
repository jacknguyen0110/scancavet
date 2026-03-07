import os
import re
import cv2
import json
import time
import base64
import logging
import requests
import numpy as np
from flask import Flask, request, jsonify

# =========================
# ENV
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

REQUEST_TIMEOUT = 120

# chống xử lý lặp
processed_update_ids = []
processed_file_ids = []
MAX_UPDATE_IDS = 5000
MAX_FILE_IDS = 3000

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# DEDUPE
# =========================
def remember_update_id(update_id):
    global processed_update_ids
    processed_update_ids.append(update_id)
    if len(processed_update_ids) > MAX_UPDATE_IDS:
        processed_update_ids = processed_update_ids[-2000:]


def remember_file_id(file_id):
    global processed_file_ids
    processed_file_ids.append(file_id)
    if len(processed_file_ids) > MAX_FILE_IDS:
        processed_file_ids = processed_file_ids[-1000:]


def is_duplicate_update(update_id):
    return update_id in processed_update_ids


def is_duplicate_file(file_id):
    return file_id in processed_file_ids


def is_stale_message(msg, max_age_seconds=180):
    msg_date = msg.get("date")
    if not msg_date:
        return False
    now_ts = int(time.time())
    age = now_ts - int(msg_date)
    return age > max_age_seconds


# =========================
# TELEGRAM
# =========================
def tg_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


def send_message(chat_id: int, text: str) -> None:
    try:
        requests.post(
            tg_url("sendMessage"),
            json={"chat_id": chat_id, "text": text},
            timeout=REQUEST_TIMEOUT
        )
    except Exception as e:
        logger.exception("send_message error: %s", e)


def get_telegram_file_url(file_id: str) -> str:
    r = requests.get(
        tg_url("getFile"),
        params={"file_id": file_id},
        timeout=REQUEST_TIMEOUT
    )
    r.raise_for_status()
    data = r.json()

    if not data.get("ok"):
        raise Exception(f"Telegram getFile lỗi: {data}")

    file_path = data["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"


def download_telegram_file(file_id: str) -> bytes:
    file_url = get_telegram_file_url(file_id)
    r = requests.get(file_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content


# =========================
# IMAGE UTILS
# =========================
def bytes_to_img(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Không decode được ảnh")
    return img


def normalize_for_model(image_bytes: bytes, max_side: int = 1800, jpg_quality: int = 90) -> bytes:
    """
    Giảm kích thước hợp lý trước khi gửi GPT Vision để nhanh và rẻ hơn.
    """
    img = bytes_to_img(image_bytes)
    h, w = img.shape[:2]
    side = max(h, w)

    if side > max_side:
        scale = max_side / float(side)
        nw = int(w * scale)
        nh = int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        raise Exception("Không encode được ảnh JPG")
    return buf.tobytes()


# =========================
# PLATE NORMALIZATION
# =========================
def normalize_plate(text: str) -> str:
    x = (text or "").upper()
    x = x.replace("O", "0")
    x = re.sub(r"[^0-9A-Z]", "", x)
    return x


def is_valid_plate(text: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[A-Z]\d{5}", text))


def postprocess_plates(plates):
    out = []
    for p in plates or []:
        n = normalize_plate(str(p))
        if is_valid_plate(n) and n not in out:
            out.append(n)
    return out


# =========================
# OPENAI VISION
# =========================
def extract_plates_with_gpt(image_bytes: bytes):
    if not OPENAI_API_KEY:
        raise Exception("Thiếu OPENAI_API_KEY")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_b64}"

    schema = {
        "name": "plate_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "plates": {
                    "type": "array",
                    "description": "Danh sách biển số xe nhìn thấy rõ trong ảnh, chuẩn hóa dạng 50H31875",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["plates"],
            "additionalProperties": False
        }
    }

    prompt = (
        "Bạn là hệ thống trích xuất biển số từ ảnh cà vẹt xe Việt Nam.\n"
        "Nhiệm vụ:\n"
        "1. Chỉ lấy biển số nhìn thấy rõ trên cà vẹt trong ảnh.\n"
        "2. Ưu tiên dòng gần nhãn 'Number Plate' hoặc 'Biển số đăng ký'.\n"
        "3. Trả biển số dạng liền không dấu gạch/chấm, ví dụ: 50H31875.\n"
        "4. Nếu ảnh có nhiều cà vẹt, trả tất cả biển số nhìn thấy rõ.\n"
        "5. Không đoán. Không bịa. Nếu không chắc thì bỏ qua.\n"
        "6. Không trả text giải thích, chỉ trả JSON đúng schema."
    )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Bạn trích xuất biển số từ ảnh và trả JSON chính xác."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema
        },
        "temperature": 0
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    if r.status_code != 200:
        raise Exception(f"OpenAI HTTP {r.status_code}: {r.text[:1000]}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
    except Exception:
        raise Exception(f"OpenAI không trả JSON hợp lệ: {content[:1000]}")

    plates = postprocess_plates(parsed.get("plates", []))
    return plates


# =========================
# APPS SCRIPT
# =========================
def send_to_apps_script(image_bytes: bytes, plates, caption: str, chat_id: int, name: str):
    if not APPS_SCRIPT_URL:
        raise Exception("Thiếu APPS_SCRIPT_URL")

    payload = {
        "image": base64.b64encode(image_bytes).decode("utf-8"),
        "plates": plates or [],
        "caption": caption or "",
        "chatId": str(chat_id),
        "name": name or ""
    }

    r = requests.post(
        APPS_SCRIPT_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True
    )

    content_type = r.headers.get("Content-Type", "")
    body_preview = r.text[:1000]

    if r.status_code != 200:
        raise Exception(f"Apps Script HTTP {r.status_code}: {body_preview}")

    is_probably_json = (
        "application/json" in content_type.lower()
        or body_preview.strip().startswith("{")
    )

    if not is_probably_json:
        raise Exception(
            f"Apps Script không trả JSON. Content-Type={content_type}. Body={body_preview}"
        )

    try:
        data = r.json()
    except Exception:
        raise Exception(f"Apps Script không trả JSON hợp lệ: {body_preview}")

    if data.get("status") != "ok":
        raise Exception(f"Apps Script lỗi: {data}")

    return data


# =========================
# MAIN
# =========================
def process_file(file_id: str, chat_id: int, full_name: str, caption: str):
    original_bytes = download_telegram_file(file_id)
    model_bytes = normalize_for_model(original_bytes)

    plates = extract_plates_with_gpt(model_bytes)
    logger.info("GPT plates: %s", plates)

    if plates:
        send_to_apps_script(
            image_bytes=original_bytes,   # lưu ảnh gốc lên Drive
            plates=plates,
            caption=caption,
            chat_id=chat_id,
            name=full_name
        )

    return plates


# =========================
# FLASK
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running"})


@app.route("/telegram/webhook", methods=["POST"])
def telegram_webhook():
    try:
        data = request.get_json(silent=True) or {}

        update_id = data.get("update_id")
        if update_id is not None and is_duplicate_update(update_id):
            logger.info("Duplicate update skipped: %s", update_id)
            return "ok", 200

        if update_id is not None:
            remember_update_id(update_id)

        msg = data.get("message") or data.get("edited_message")
        if not msg:
            return "ok", 200

        if is_stale_message(msg, max_age_seconds=180):
            logger.info("Stale message skipped: date=%s", msg.get("date"))
            return "ok", 200

        chat_id = msg["chat"]["id"]
        user = msg.get("from", {}) or {}
        full_name = f'{user.get("first_name", "")} {user.get("last_name", "")}'.strip()
        caption = msg.get("caption", "") or ""

        file_id = None

        if msg.get("photo"):
            file_id = msg["photo"][-1]["file_id"]

        elif msg.get("document"):
            doc = msg["document"]
            mime = (doc.get("mime_type") or "").lower()
            fname = (doc.get("file_name") or "").lower()
            is_image = mime.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp"))

            if not is_image:
                send_message(chat_id, "⚠️ File này không phải ảnh. Hãy gửi JPG/PNG/WebP.")
                return "ok", 200

            file_id = doc["file_id"]

        else:
            send_message(chat_id, "📸 Gửi ảnh cà vẹt để quét biển số.")
            return "ok", 200

        if file_id and is_duplicate_file(file_id):
            logger.info("Duplicate file skipped: %s", file_id)
            return "ok", 200

        if file_id:
            remember_file_id(file_id)

        plates = process_file(file_id, chat_id, full_name, caption)

        if plates:
            send_message(
                chat_id,
                "✅ Đã quét và lưu thành công.\n"
                f"Số biển số đọc được: {len(plates)}\n"
                "Biển số:\n" + "\n".join(plates)
            )
        else:
            send_message(
                chat_id,
                "⚠️ Chưa đọc chắc chắn ra biển số nên không ghi vào Drive/Sheet."
            )

        return "ok", 200

    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)

        try:
            data = request.get_json(silent=True) or {}
            msg = data.get("message") or {}
            chat_id = msg.get("chat", {}).get("id")
            if chat_id:
                send_message(chat_id, f"❌ Lỗi xử lý: {str(e)}")
        except Exception:
            pass

        return "ok", 200


if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN")
    if not OPENAI_API_KEY:
        raise RuntimeError("Thiếu OPENAI_API_KEY")

    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
