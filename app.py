import os
import re
import cv2
import base64
import logging
import requests
import numpy as np
import pytesseract
from flask import Flask, request, jsonify

# =========================
# CONFIG
# =========================
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


def resize_for_processing(img: np.ndarray, max_side: int = 2600):
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
# GRID SPLIT
# =========================
def split_cards_grid_5x4(img: np.ndarray):
    """
    Ảnh collage 20 cà vẹt: chia cố định 5 hàng x 4 cột.
    """
    h, w = img.shape[:2]
    rows, cols = 5, 4

    cell_h = h / rows
    cell_w = w / cols
    crops = []

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            # cắt bớt mép để tránh viền trắng
            pad_x = int((x2 - x1) * 0.04)
            pad_y = int((y2 - y1) * 0.04)

            x1 = max(0, x1 + pad_x)
            y1 = max(0, y1 + pad_y)
            x2 = min(w, x2 - pad_x)
            y2 = min(h, y2 - pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    return crops


def looks_like_collage(img: np.ndarray) -> bool:
    """
    Nhận diện ảnh nhiều cà vẹt theo kích thước và tỷ lệ.
    """
    h, w = img.shape[:2]
    ratio = w / float(h) if h else 0

    # ảnh bạn gửi gần kiểu 4 cột x 5 hàng
    return (w >= 900 and h >= 1200 and 0.65 <= ratio <= 0.95)


def detect_cards(img: np.ndarray):
    """
    - Nếu là ảnh collage: chia 5x4
    - Nếu không: coi là 1 cà vẹt
    """
    if looks_like_collage(img):
        crops = split_cards_grid_5x4(img)
        if len(crops) == 20:
            return crops

    return [img]


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
    29A-111.11 -> 29A11111
    50F-054.23 -> 50F05423
    """
    x = text.upper()
    x = x.replace("O", "0")
    x = re.sub(r"[^0-9A-Z]", "", x)
    return x


def is_valid_plate_compact(text: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[A-Z]\d{5}", text))


def extract_plate_after_label(text: str):
    """
    Ưu tiên lấy biển số sau cụm Number Plate / Biển số.
    """
    raw = clean_text(text)

    patterns = [
        r"NUMBER\s*PLATE[^A-Z0-9]{0,50}(\d{2}[A-Z]-?\d{3}[.\s]?\d{2})",
        r"BIEN\s*SO[^A-Z0-9]{0,50}(\d{2}[A-Z]-?\d{3}[.\s]?\d{2})",
        r"BIỂN\s*SỐ[^A-Z0-9]{0,50}(\d{2}[A-Z]-?\d{3}[.\s]?\d{2})",
    ]

    for p in patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            plate = normalize_plate_to_compact(m.group(1))
            if is_valid_plate_compact(plate):
                return plate

    return None


def extract_any_plate(text: str):
    raw = clean_text(text)

    patterns = [
        r"\b\d{2}[A-Z]-\d{3}\.\d{2}\b",
        r"\b\d{2}[A-Z]\d{5}\b",
        r"\b\d{2}[A-Z]\s?\d{3}\s?\d{2}\b",
        r"\b\d{2}[A-Z]-?\d{3}[.\s]?\d{2}\b",
    ]

    for p in patterns:
        matches = re.findall(p, raw)
        for item in matches:
            plate = normalize_plate_to_compact(item)
            if is_valid_plate_compact(plate):
                return plate

    return None


def extract_plate_from_crop(card_img: np.ndarray):
    """
    Chỉ trả về 1 biển số tốt nhất cho mỗi crop.
    """
    h, w = card_img.shape[:2]
    rois = []

    # ROI vùng biển số
    roi1 = card_img[int(h * 0.42):int(h * 0.95), 0:int(w * 0.70)]
    if roi1.size > 0:
        rois.append(roi1)

    # ROI rộng hơn
    roi2 = card_img[int(h * 0.35):int(h * 0.98), 0:int(w * 0.85)]
    if roi2.size > 0:
        rois.append(roi2)

    # fallback toàn thẻ
    rois.append(card_img)

    all_texts = []

    for roi in rois:
        txt1 = pytesseract.image_to_string(
            roi,
            lang=TESS_LANG,
            config="--oem 3 --psm 6"
        )
        all_texts.append(txt1)

        txt2 = pytesseract.image_to_string(
            preprocess_for_ocr(roi),
            lang=TESS_LANG,
            config="--oem 3 --psm 6"
        )
        all_texts.append(txt2)

        combined = "\n".join(all_texts)

        plate = extract_plate_after_label(combined)
        if plate:
            return plate

        plate = extract_any_plate(combined)
        if plate:
            return plate

    return None


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
    img, _ = resize_for_processing(img, max_side=2600)

    crops = detect_cards(img)
    logger.info("Detected crops: %s", len(crops))

    found_plates = []
    saved_count = 0

    for idx, crop in enumerate(crops, start=1):
        try:
            plate = extract_plate_from_crop(crop)
            logger.info("Crop %s plate: %s", idx, plate)

            if not plate:
                continue

            if plate in found_plates:
                continue

            crop_bytes = img_to_jpg_bytes(crop)

            send_to_apps_script(
                img_bytes=crop_bytes,
                plate=plate,
                caption=caption,
                chat_id=chat_id,
                name=full_name
            )

            found_plates.append(plate)
            saved_count += 1

        except Exception as e:
            logger.exception("Crop %s error: %s", idx, e)

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
