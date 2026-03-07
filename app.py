import os
import re
import cv2
import base64
import logging
import requests
import numpy as np
import pytesseract
from flask import Flask, request, jsonify

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL", "").strip()
TESS_LANG = os.getenv("TESS_LANG", "vie+eng").strip()

REQUEST_TIMEOUT = 120

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
# IMAGE
# =========================
def bytes_to_img(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Không decode được ảnh đầu vào")
    return img


def img_to_jpg_bytes(img: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise Exception("Không encode được ảnh JPG")
    return buf.tobytes()


def resize_for_processing(img: np.ndarray, max_side: int = 1800):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img.copy(), 1.0

    scale = max_side / float(side)
    nw = int(w * scale)
    nh = int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale


# =========================
# DETECT CARD
# =========================
def detect_cards(img: np.ndarray):
    original = img.copy()
    work, scale = resize_for_processing(img, max_side=1800)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = work.shape[:2]
    min_area = (h_img * w_img) * 0.02

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h else 0

        if 1.2 <= ratio <= 2.5 and w > 250 and h > 120:
            ox = int(x / scale)
            oy = int(y / scale)
            ow = int(w / scale)
            oh = int(h / scale)

            ox = max(0, ox)
            oy = max(0, oy)
            ow = min(original.shape[1] - ox, ow)
            oh = min(original.shape[0] - oy, oh)

            crop = original[oy:oy + oh, ox:ox + ow]
            if crop.size > 0:
                candidates.append((ox, oy, crop))

    if not candidates:
        return [original]

    candidates.sort(key=lambda x: (x[1], x[0]))

    results = []
    seen = set()
    for _, _, crop in candidates:
        h, w = crop.shape[:2]
        key = (round(w / 50), round(h / 50))
        if key in seen:
            continue
        seen.add(key)
        results.append(crop)

    return results[:20]


# =========================
# OCR
# =========================
def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )
    return th


def clean_text(text: str) -> str:
    if not text:
        return ""
    x = text.upper()
    x = x.replace("–", "-").replace("—", "-")
    x = x.replace(",", ".")
    return x


def normalize_plate_to_compact(text: str) -> str:
    """
    50F-054.23 -> 50F05423
    """
    x = text.upper()
    x = x.replace("O", "0")
    x = re.sub(r"[^0-9A-Z]", "", x)
    return x


def is_valid_plate_compact(text: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[A-Z]\d{5}", text))


def extract_plate_after_number_plate_label(text: str):
    """
    Ưu tiên bắt biển số nằm ngay sau cụm Number Plate / Biển số đăng ký.
    """
    raw = clean_text(text)

    patterns = [
        r"NUMBER\s*PLATE[^A-Z0-9]{0,30}(\d{2}[A-Z]\s?-?\s?\d{3}[.\s]?\d{2})",
        r"BIEN\s*SO[^A-Z0-9]{0,30}(\d{2}[A-Z]\s?-?\s?\d{3}[.\s]?\d{2})",
        r"BIỂN\s*SỐ[^A-Z0-9]{0,30}(\d{2}[A-Z]\s?-?\s?\d{3}[.\s]?\d{2})",
    ]

    for p in patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            plate = normalize_plate_to_compact(m.group(1))
            if is_valid_plate_compact(plate):
                return plate

    return None


def extract_any_valid_plate(text: str):
    raw = clean_text(text)

    patterns = [
        r"\b\d{2}[A-Z]-?\d{3}\.?\d{2}\b",
        r"\b\d{2}[A-Z]\d{5}\b",
        r"\b\d{2}[A-Z]\s?\d{3}\s?\d{2}\b",
    ]

    found = []
    for p in patterns:
        found.extend(re.findall(p, raw))

    for item in found:
        plate = normalize_plate_to_compact(item)
        if is_valid_plate_compact(plate):
            return plate

    return None


def extract_plate_from_crop(card_img: np.ndarray):
    """
    Chỉ trả về 1 biển số tốt nhất.
    Không trả full OCR.
    """
    h, w = card_img.shape[:2]
    rois = []

    roi1 = card_img[int(h * 0.45):int(h * 0.92), 0:int(w * 0.62)]
    if roi1.size > 0:
        rois.append(roi1)

    roi2 = card_img[int(h * 0.40):int(h * 0.90), int(w * 0.03):int(w * 0.75)]
    if roi2.size > 0:
        rois.append(roi2)

    rois.append(card_img)

    collected_texts = []

    for roi in rois:
        txt1 = pytesseract.image_to_string(
            roi,
            lang=TESS_LANG,
            config="--oem 3 --psm 6"
        )
        collected_texts.append(txt1)

        txt2 = pytesseract.image_to_string(
            preprocess_for_ocr(roi),
            lang=TESS_LANG,
            config="--oem 3 --psm 6"
        )
        collected_texts.append(txt2)

        combined = "\n".join(collected_texts)

        # Ưu tiên đúng nhãn Number Plate
        plate = extract_plate_after_number_plate_label(combined)
        if plate:
            return plate

    # fallback cuối cùng
    combined = "\n".join(collected_texts)
    return extract_any_valid_plate(combined)


# =========================
# APPS SCRIPT
# =========================
def send_to_apps_script(
    img_bytes: bytes,
    plate: str,
    caption: str,
    chat_id: int,
    name: str
):
    if not APPS_SCRIPT_URL:
        raise Exception("Thiếu APPS_SCRIPT_URL")

    payload = {
        "image": base64.b64encode(img_bytes).decode("utf-8"),
        "plate": plate or "",
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
    raw_bytes = download_telegram_file(file_id)
    img = bytes_to_img(raw_bytes)
    crops = detect_cards(img)

    found_plates = []
    saved_count = 0

    for crop in crops:
        plate = extract_plate_from_crop(crop)

        # chỉ lưu crop có đọc ra biển số
        if not plate:
            continue

        crop_bytes = img_to_jpg_bytes(crop)

        send_to_apps_script(
            img_bytes=crop_bytes,
            plate=plate,
            caption=caption,
            chat_id=chat_id,
            name=full_name
        )

        saved_count += 1
        found_plates.append(plate)

    found_plates = list(dict.fromkeys(found_plates))
    return found_plates, saved_count


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
        msg = data.get("message") or data.get("edited_message")

        if not msg:
            return "ok", 200

        chat_id = msg["chat"]["id"]
        user = msg.get("from", {}) or {}
        full_name = f'{user.get("first_name", "")} {user.get("last_name", "")}'.strip()
        caption = msg.get("caption", "") or ""

        if msg.get("photo"):
            file_id = msg["photo"][-1]["file_id"]
            plates, saved_count = process_file(file_id, chat_id, full_name, caption)

        elif msg.get("document"):
            doc = msg["document"]
            mime = (doc.get("mime_type") or "").lower()
            fname = (doc.get("file_name") or "").lower()
            is_image = mime.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp"))

            if not is_image:
                send_message(chat_id, "⚠️ File này không phải ảnh. Hãy gửi JPG/PNG/WebP.")
                return "ok", 200

            file_id = doc["file_id"]
            plates, saved_count = process_file(file_id, chat_id, full_name, caption)

        else:
            send_message(chat_id, "📸 Gửi ảnh cà vẹt để quét biển số.")
            return "ok", 200

        if plates:
            send_message(
                chat_id,
                "✅ Đã quét và lưu thành công.\n"
                f"Số ảnh crop đã lưu: {saved_count}\n"
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

    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
