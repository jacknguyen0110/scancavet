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

# Regex chuẩn biển số dạng:
# 29A-111.11 / 50F-054.23 / 31H-444.44
PLATE_REGEXES = [
    r"\b\d{2}[A-Z]-\d{3}\.\d{2}\b",
    r"\b\d{2}[A-Z]\d{5}\b",
    r"\b\d{2}[A-Z]\s?\d{3}\s?\d{2}\b",
    r"\b\d{2}[A-Z]-?\d{3}[.\s]?\d{2}\b",
]

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


def resize_for_processing(img: np.ndarray, max_side: int = 2200):
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
# CARD SPLIT FOR MULTI-CARD IMAGE
# =========================
def has_card_like_text(crop: np.ndarray) -> bool:
    """
    OCR nhanh vùng trên/trái của crop để kiểm tra có phải cà vẹt không.
    """
    h, w = crop.shape[:2]
    if h < 120 or w < 220:
        return False

    roi = crop[0:int(h * 0.45), 0:int(w * 0.75)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(gray, lang=TESS_LANG, config="--oem 3 --psm 6")
    up = txt.upper()

    keywords = [
        "OWNER", "NUMBER PLATE", "BIEN SO", "VINFAST",
        "CHASSIS", "ENGINE", "CTCP", "GSM"
    ]
    score = sum(1 for k in keywords if k in up)
    return score >= 2


def split_cards_grid(img: np.ndarray):
    """
    Ảnh nhiều cà vẹt thường theo lưới.
    Thử các cấu hình lưới phổ biến rồi lọc bằng OCR nhẹ.
    """
    h, w = img.shape[:2]

    # Ưu tiên cấu hình gần đúng với ảnh bạn gửi
    grid_candidates = [
        (5, 4),  # 20 thẻ
        (4, 4),  # 16 thẻ
        (5, 5),
        (4, 5),
        (6, 4),
        (3, 4),
    ]

    best_crops = []
    best_score = -1

    for rows, cols in grid_candidates:
        cell_h = h / rows
        cell_w = w / cols
        temp = []
        score = 0

        for r in range(rows):
            for c in range(cols):
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)

                # cắt bớt mép để tránh viền trắng
                pad_x = int((x2 - x1) * 0.03)
                pad_y = int((y2 - y1) * 0.03)

                x1p = min(max(0, x1 + pad_x), w)
                y1p = min(max(0, y1 + pad_y), h)
                x2p = min(max(0, x2 - pad_x), w)
                y2p = min(max(0, y2 - pad_y), h)

                crop = img[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                ok = has_card_like_text(crop)
                if ok:
                    score += 1
                temp.append((r, c, crop, ok))

        if score > best_score:
            best_score = score
            best_crops = temp

    valid = [x[2] for x in best_crops if x[3]]

    # fallback nếu OCR check chưa nhận được nhiều
    if len(valid) >= 4:
        return valid

    # fallback 2: dùng grid 5x4 luôn
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

            pad_x = int((x2 - x1) * 0.03)
            pad_y = int((y2 - y1) * 0.03)

            x1p = min(max(0, x1 + pad_x), w)
            y1p = min(max(0, y1 + pad_y), h)
            x2p = min(max(0, x2 - pad_x), w)
            y2p = min(max(0, y2 - pad_y), h)

            crop = img[y1p:y2p, x1p:x2p]
            if crop.size > 0:
                crops.append(crop)

    return crops


def detect_cards(img: np.ndarray):
    """
    Với ảnh nhiều cà vẹt: dùng grid split.
    Với ảnh 1 cà vẹt: trả nguyên ảnh.
    """
    h, w = img.shape[:2]

    # Nếu ảnh đủ lớn và có vẻ là collage nhiều thẻ thì split grid
    if w > 1400 and h > 1800:
        crops = split_cards_grid(img)
        if len(crops) >= 4:
            return crops

    return [img]


# =========================
# OCR / PLATE
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
    """
    x = text.upper()
    x = x.replace("O", "0")
    x = re.sub(r"[^0-9A-Z]", "", x)
    return x


def is_valid_plate_compact(text: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[A-Z]\d{5}", text))


def extract_plate_candidates(text: str):
    raw = clean_text(text)
    out = []

    for pattern in PLATE_REGEXES:
        matches = re.findall(pattern, raw)
        for m in matches:
            plate = normalize_plate_to_compact(m)
            if is_valid_plate_compact(plate) and plate not in out:
                out.append(plate)

    return out


def extract_plate_after_label(text: str):
    raw = clean_text(text)

    patterns = [
        r"NUMBER\s*PLATE[^A-Z0-9]{0,40}(\d{2}[A-Z]-?\d{3}[.\s]?\d{2})",
        r"BIEN\s*SO[^A-Z0-9]{0,40}(\d{2}[A-Z]-?\d{3}[.\s]?\d{2})",
        r"BIỂN\s*SỐ[^A-Z0-9]{0,40}(\d{2}[A-Z]-?\d{3}[.\s]?\d{2})",
    ]

    for p in patterns:
        m = re.search(p, raw, flags=re.IGNORECASE)
        if m:
            plate = normalize_plate_to_compact(m.group(1))
            if is_valid_plate_compact(plate):
                return plate

    return None


def extract_plate_from_crop(card_img: np.ndarray):
    """
    Chỉ trả về 1 biển số tốt nhất cho mỗi thẻ.
    """
    h, w = card_img.shape[:2]
    rois = []

    # ROI biển số - nửa dưới bên trái
    roi1 = card_img[int(h * 0.46):int(h * 0.93), 0:int(w * 0.63)]
    if roi1.size > 0:
        rois.append(roi1)

    # ROI rộng hơn một chút
    roi2 = card_img[int(h * 0.38):int(h * 0.95), int(w * 0.02):int(w * 0.80)]
    if roi2.size > 0:
        rois.append(roi2)

    # fallback toàn thẻ
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

        plate = extract_plate_after_label(combined)
        if plate:
            return plate

        candidates = extract_plate_candidates(combined)
        if candidates:
            return candidates[0]

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
    img, _ = resize_for_processing(img, max_side=2200)

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

            saved_count += 1
            found_plates.append(plate)

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
