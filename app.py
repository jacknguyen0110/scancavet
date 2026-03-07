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
# CARD DETECTION
# =========================
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 50 or max_height < 50:
        return None

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


def detect_rect_cards(img: np.ndarray):
    original = img.copy()
    h0, w0 = original.shape[:2]

    work, scale = resize_for_processing(original, max_side=1800)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (work.shape[0] * work.shape[1]) * 0.02
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            pts[:, 0] /= scale
            pts[:, 1] /= scale

            warped = four_point_transform(original, pts)
            if warped is None:
                continue

            h, w = warped.shape[:2]
            ratio = w / float(h) if h else 0

            if 1.10 <= ratio <= 2.6 and w >= 220 and h >= 120:
                x, y, _, _ = cv2.boundingRect(approx)
                candidates.append((x, y, warped))
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / float(h) if h else 0

            if 1.10 <= ratio <= 2.6 and w >= 200 and h >= 120:
                ox = int(x / scale)
                oy = int(y / scale)
                ow = int(w / scale)
                oh = int(h / scale)

                ox = max(0, ox)
                oy = max(0, oy)
                ow = min(w0 - ox, ow)
                oh = min(h0 - oy, oh)

                crop = original[oy:oy + oh, ox:ox + ow]
                if crop.size > 0:
                    candidates.append((ox, oy, crop))

    if not candidates:
        return []

    candidates.sort(key=lambda t: (t[1], t[0]))

    results = []
    seen = set()

    for _, _, crop in candidates:
        h, w = crop.shape[:2]
        key = (round(w / 35), round(h / 35))
        if key in seen:
            continue
        seen.add(key)
        results.append(crop)

    return results


def looks_like_collage(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    ratio = w / float(h) if h else 0
    return w >= 700 and h >= 900 and 0.65 <= ratio <= 1.05


def split_cards_grid_5x4(img: np.ndarray):
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

            pad_x = int((x2 - x1) * 0.02)
            pad_y = int((y2 - y1) * 0.02)

            x1 = max(0, x1 + pad_x)
            y1 = max(0, y1 + pad_y)
            x2 = min(w, x2 - pad_x)
            y2 = min(h, y2 - pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    return crops


def looks_like_vertical_stack(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    ratio = h / float(w) if w else 0
    return h >= 1000 and w >= 500 and ratio >= 1.5


def split_vertical_stack(img: np.ndarray):
    """
    Chia ảnh xếp dọc 3-6 cà vẹt.
    Dùng projection theo trục Y để tìm khoảng trắng giữa các thẻ.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # thẻ sáng, nền bàn gỗ trung bình -> threshold để lấy vùng thẻ
    _, th = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    # tính số pixel trắng theo từng dòng
    proj = np.sum(th == 255, axis=1)

    # dòng thuộc vùng thẻ nếu đủ trắng
    white_threshold = int(w * 0.35)
    mask = proj > white_threshold

    bands = []
    in_band = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_band:
            start = i
            in_band = True
        elif not val and in_band:
            end = i
            if end - start > 120:
                bands.append((start, end))
            in_band = False

    if in_band:
        end = h
        if end - start > 120:
            bands.append((start, end))

    crops = []
    for y1, y2 in bands:
        pad_y = int((y2 - y1) * 0.03)
        yy1 = max(0, y1 - pad_y)
        yy2 = min(h, y2 + pad_y)

        # tìm biên ngang hữu ích
        band = img[yy1:yy2, :]
        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        _, th_band = cv2.threshold(gray_band, 180, 255, cv2.THRESH_BINARY)

        proj_x = np.sum(th_band == 255, axis=0)
        xmask = proj_x > int((yy2 - yy1) * 0.20)

        xs = np.where(xmask)[0]
        if len(xs) == 0:
            continue

        x1 = max(0, int(xs.min()) - 10)
        x2 = min(w, int(xs.max()) + 10)

        crop = img[yy1:yy2, x1:x2]
        if crop.size == 0:
            continue

        ch, cw = crop.shape[:2]
        ratio = cw / float(ch) if ch else 0
        if 1.10 <= ratio <= 2.6:
            crops.append(crop)

    return crops


def detect_cards(img: np.ndarray):
    h, w = img.shape[:2]
    ratio = w / float(h) if h else 0
    logger.info("Image size: w=%s h=%s ratio=%.3f", w, h, ratio)

    rect_cards = detect_rect_cards(img)
    if len(rect_cards) >= 2:
        logger.info("Rect-card detect -> %s crops", len(rect_cards))
        return rect_cards

    if looks_like_vertical_stack(img):
        vertical_cards = split_vertical_stack(img)
        if len(vertical_cards) >= 2:
            logger.info("Vertical-stack detect -> %s crops", len(vertical_cards))
            return vertical_cards

    if looks_like_collage(img):
        crops = split_cards_grid_5x4(img)
        logger.info("Grid split 5x4 -> %s crops", len(crops))
        return crops

    logger.info("Single-card mode")
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
    x = text.upper()
    x = x.replace("O", "0")
    x = re.sub(r"[^0-9A-Z]", "", x)
    return x


def is_valid_plate_compact(text: str) -> bool:
    return bool(re.fullmatch(r"\d{2}[A-Z]\d{5}", text))


def extract_plate_after_label(text: str):
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
    h, w = card_img.shape[:2]
    rois = []

    roi1 = card_img[int(h * 0.42):int(h * 0.95), 0:int(w * 0.70)]
    if roi1.size > 0:
        rois.append(roi1)

    roi2 = card_img[int(h * 0.35):int(h * 0.98), 0:int(w * 0.85)]
    if roi2.size > 0:
        rois.append(roi2)

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
def send_to_apps_script(img_bytes: bytes, plate: str, caption: str, chat_id: int, name: str):
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
            if plate:
                logger.info("Crop %s OK: %s", idx, plate)

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

        update_id = data.get("update_id")
        if update_id is not None and is_duplicate_update(update_id):
            logger.info("Duplicate update skipped: %s", update_id)
            return "ok", 200

        if update_id is not None:
            remember_update_id(update_id)

        msg = data.get("message") or data.get("edited_message")
        if not msg:
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

        plates, saved_count = process_file(file_id, chat_id, full_name, caption)

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
