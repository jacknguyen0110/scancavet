import os
import re
import io
import json
import uuid
import logging
from datetime import datetime

import cv2
import numpy as np
import requests
import pytesseract
from flask import Flask, request, jsonify

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_SECRET_TOKEN = os.getenv("TELEGRAM_SECRET_TOKEN", "").strip()

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1").strip()

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "").strip()

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

# Tesseract language
TESS_LANG = os.getenv("TESS_LANG", "eng")
# Nếu Docker có cài vie + eng thì để "vie+eng"

# OCR config
TESS_CONFIG = r'--oem 3 --psm 6'

# App
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# GOOGLE AUTH
# =========================
def get_google_credentials():
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON")

    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    return Credentials.from_service_account_info(info, scopes=scopes)


def get_gspread_client():
    creds = get_google_credentials()
    return gspread.authorize(creds)


def get_drive_service():
    creds = get_google_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# =========================
# SHEET
# =========================
def get_sheet():
    gc = get_gspread_client()
    sh = gc.open_by_key(SPREADSHEET_ID)
    ws = sh.worksheet(SHEET_NAME)
    return ws


def ensure_headers():
    ws = get_sheet()
    values = ws.get_all_values()
    headers = [
        "timestamp",
        "chat_id",
        "full_name",
        "caption",
        "plate_number",
        "drive_file_url",
        "drive_preview_formula",
        "ocr_text",
        "status",
        "source_telegram_file_id",
        "crop_index",
        "raw_image_width",
        "raw_image_height",
    ]
    if not values:
        ws.append_row(headers)
        return

    first_row = values[0]
    if [x.strip() for x in first_row[:len(headers)]] != headers:
        # chỉ thêm header nếu hàng đầu đang trống
        if "".join(first_row).strip() == "":
            ws.update("A1:M1", [headers])


def append_sheet_row(row):
    ws = get_sheet()
    ws.append_row(row, value_input_option="USER_ENTERED")


# =========================
# DRIVE
# =========================
def upload_bytes_to_drive(file_bytes: bytes, filename: str, mimetype: str = "image/jpeg"):
    drive = get_drive_service()
    media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mimetype, resumable=False)

    file_metadata = {
        "name": filename,
        "parents": [DRIVE_FOLDER_ID]
    }

    created = drive.files().create(
        body=file_metadata,
        media_body=media,
        fields="id,name,webViewLink,webContentLink"
    ).execute()

    file_id = created["id"]

    # share anyone with link can view
    try:
        drive.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"}
        ).execute()
    except Exception as e:
        logger.warning(f"Cannot set public permission for file {file_id}: {e}")

    web_view = created.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}/view"
    return file_id, web_view


def make_drive_preview_formula(file_id: str):
    return f'=IMAGE("https://drive.google.com/thumbnail?id={file_id}&sz=w300",1)'


# =========================
# TELEGRAM
# =========================
def telegram_api_url(method: str):
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def get_telegram_file_url(file_id: str):
    url = telegram_api_url("getFile")
    resp = requests.get(url, params={"file_id": file_id}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise ValueError(f"Telegram getFile failed: {data}")
    file_path = data["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"


def send_telegram_message(chat_id: int, text: str):
    url = telegram_api_url("sendMessage")
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    try:
        requests.post(url, json=payload, timeout=60)
    except Exception as e:
        logger.warning(f"send_telegram_message error: {e}")


def set_telegram_webhook(base_url: str):
    url = telegram_api_url("setWebhook")
    payload = {
        "url": f"{base_url.rstrip('/')}/telegram/webhook"
    }
    if TELEGRAM_SECRET_TOKEN:
        payload["secret_token"] = TELEGRAM_SECRET_TOKEN

    resp = requests.post(url, json=payload, timeout=60)
    return resp.json()


# =========================
# IMAGE UTIL
# =========================
def download_image_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def bytes_to_cv2_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return image


def cv2_image_to_jpg_bytes(img, quality=95):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Cannot encode image to jpg")
    return buf.tobytes()


def resize_for_processing(img, max_side=1600):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img.copy(), 1.0
    scale = max_side / side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

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

    if max_width < 10 or max_height < 10:
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


# =========================
# DETECT CÀ VẸT
# =========================
def detect_card_crops(image):
    """
    Trả về list ảnh crop từng cà vẹt.
    Ưu tiên tìm contour 4 cạnh có diện tích đủ lớn.
    """
    original = image.copy()
    resized, scale = resize_for_processing(image, max_side=1800)
    rh, rw = resized.shape[:2]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    img_area = rh * rw

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.02:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")

            # scale ngược lên ảnh gốc
            pts[:, 0] /= scale
            pts[:, 1] /= scale

            warped = four_point_transform(original, pts)
            if warped is None:
                continue

            h, w = warped.shape[:2]
            ratio = w / float(h) if h > 0 else 0

            # cà vẹt ngang, gần giống thẻ
            if 1.2 <= ratio <= 2.4 and w > 300 and h > 150:
                candidates.append((area, warped))

    # fallback: nếu không detect được contour chuẩn, lấy toàn ảnh
    if not candidates:
        return [original]

    # sort lớn -> nhỏ
    candidates.sort(key=lambda x: x[0], reverse=True)

    # remove near-duplicate by size
    result = []
    seen = []
    for _, crop in candidates:
        h, w = crop.shape[:2]
        key = (round(w / 50), round(h / 50))
        if key in seen:
            continue
        seen.append(key)
        result.append(crop)

    return result[:20]


# =========================
# OCR / PLATE
# =========================
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # tăng nét
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    return th


def normalize_plate(text: str):
    if not text:
        return None

    x = text.upper()
    x = x.replace(" ", "")
    x = x.replace(",", ".")
    x = x.replace("O", "0")
    x = x.replace("I", "1")
    x = x.replace("L", "1") if re.match(r'^\d{2}[A-Z]', x) is None else x
    x = re.sub(r"[^0-9A-Z\.-]", "", x)

    patterns = [
        r"^(\d{2})([A-Z])(\d{3})(\d{2})$",
        r"^(\d{2})([A-Z])-(\d{3})(\d{2})$",
        r"^(\d{2})([A-Z])(\d{3})\.(\d{2})$",
        r"^(\d{2})([A-Z])-(\d{3})\.(\d{2})$",
    ]

    for p in patterns:
        m = re.match(p, x)
        if m:
            return f"{m.group(1)}{m.group(2)}-{m.group(3)}.{m.group(4)}"

    return None


def extract_plate_candidates(text: str):
    if not text:
        return []

    raw = text.upper()
    raw = raw.replace("–", "-").replace("—", "-")
    raw = raw.replace(",", ".")
    raw = raw.replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw)

    patterns = [
        r"\b\d{2}[A-Z]-?\d{3}[.\s]?\d{2}\b",
        r"\b\d{2}[A-Z]\s?\d{3}\s?\d{2}\b",
        r"\b\d{2}[A-Z]-?\d{5}\b",
    ]

    found = []
    for p in patterns:
        found.extend(re.findall(p, raw))

    normalized = []
    for item in found:
        plate = normalize_plate(item)
        if plate and plate not in normalized:
            normalized.append(plate)

    return normalized


def extract_plate_from_crop(crop):
    """
    OCR nhiều cách để tăng xác suất.
    """
    texts = []

    # OCR ảnh gốc
    txt1 = pytesseract.image_to_string(crop, lang=TESS_LANG, config=TESS_CONFIG)
    texts.append(txt1)

    # OCR ảnh threshold
    pre = preprocess_for_ocr(crop)
    txt2 = pytesseract.image_to_string(pre, lang=TESS_LANG, config=TESS_CONFIG)
    texts.append(txt2)

    # OCR vùng dưới bên trái (thường là chỗ biển số)
    h, w = crop.shape[:2]
    roi = crop[int(h * 0.45):h, 0:int(w * 0.65)]
    if roi.size > 0:
        txt3 = pytesseract.image_to_string(roi, lang=TESS_LANG, config=TESS_CONFIG)
        texts.append(txt3)

        pre_roi = preprocess_for_ocr(roi)
        txt4 = pytesseract.image_to_string(pre_roi, lang=TESS_LANG, config=TESS_CONFIG)
        texts.append(txt4)

    combined = "\n".join(texts)
    plates = extract_plate_candidates(combined)
    return plates, combined


# =========================
# MAIN PROCESS
# =========================
def process_telegram_image(file_id: str, chat_id: int, full_name: str, caption: str):
    ensure_headers()

    file_url = get_telegram_file_url(file_id)
    file_bytes = download_image_bytes(file_url)
    image = bytes_to_cv2_image(file_bytes)
    h, w = image.shape[:2]

    crops = detect_card_crops(image)

    results = []
    for idx, crop in enumerate(crops, start=1):
        plates, ocr_text = extract_plate_from_crop(crop)

        crop_bytes = cv2_image_to_jpg_bytes(crop, quality=95)
        filename = f"cavet_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}_{uuid.uuid4().hex[:8]}.jpg"
        drive_file_id, drive_url = upload_bytes_to_drive(crop_bytes, filename, "image/jpeg")
        thumb_formula = make_drive_preview_formula(drive_file_id)

        if not plates:
            append_sheet_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                str(chat_id),
                full_name,
                caption,
                "",
                drive_url,
                thumb_formula,
                ocr_text,
                "NO_PLATE",
                file_id,
                idx,
                w,
                h,
            ])
            results.append({
                "crop_index": idx,
                "plate": None,
                "drive_url": drive_url
            })
        else:
            for plate in plates:
                append_sheet_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    str(chat_id),
                    full_name,
                    caption,
                    plate,
                    drive_url,
                    thumb_formula,
                    ocr_text,
                    "OK",
                    file_id,
                    idx,
                    w,
                    h,
                ])
                results.append({
                    "crop_index": idx,
                    "plate": plate,
                    "drive_url": drive_url
                })

    return results


# =========================
# FLASK ROUTES
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "ok": True,
        "message": "Telegram Cavet OCR bot is running"
    })


@app.route("/set-webhook", methods=["GET"])
def route_set_webhook():
    railway_public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
    if not railway_public_domain:
        return jsonify({"ok": False, "error": "Missing RAILWAY_PUBLIC_DOMAIN env"}), 400

    if not railway_public_domain.startswith("http"):
        base_url = f"https://{railway_public_domain}"
    else:
        base_url = railway_public_domain

    result = set_telegram_webhook(base_url)
    return jsonify(result)


@app.route("/telegram/webhook", methods=["POST"])
def telegram_webhook():
    # verify secret token if configured
    if TELEGRAM_SECRET_TOKEN:
        header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if header_secret != TELEGRAM_SECRET_TOKEN:
            return jsonify({"ok": False, "error": "Invalid secret token"}), 403

    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message")

    if not message:
        return jsonify({"ok": True, "message": "No message"}), 200

    chat_id = message.get("chat", {}).get("id")
    from_user = message.get("from", {}) or {}
    full_name = f"{from_user.get('first_name', '')} {from_user.get('last_name', '')}".strip()
    caption = message.get("caption", "") or ""

    try:
        if message.get("photo"):
            file_id = message["photo"][-1]["file_id"]
            results = process_telegram_image(file_id, chat_id, full_name, caption)

            ok_plates = [x["plate"] for x in results if x["plate"]]
            if ok_plates:
                msg = "✅ Đã xử lý xong.\n" + "\n".join([f"- {p}" for p in ok_plates[:20]])
            else:
                msg = "⚠️ Đã crop và lưu ảnh, nhưng chưa đọc ra biển số."
            send_telegram_message(chat_id, msg)
            return jsonify({"ok": True}), 200

        elif message.get("document"):
            doc = message["document"]
            mime = (doc.get("mime_type") or "").lower()
            filename = (doc.get("file_name") or "").lower()
            is_image = mime.startswith("image/") or filename.endswith((".jpg", ".jpeg", ".png", ".webp"))

            if not is_image:
                send_telegram_message(chat_id, "⚠️ File này không phải ảnh. Hãy gửi JPG/PNG/WebP.")
                return jsonify({"ok": True}), 200

            file_id = doc["file_id"]
            results = process_telegram_image(file_id, chat_id, full_name, caption)

            ok_plates = [x["plate"] for x in results if x["plate"]]
            if ok_plates:
                msg = "✅ Đã xử lý xong.\n" + "\n".join([f"- {p}" for p in ok_plates[:20]])
            else:
                msg = "⚠️ Đã crop và lưu ảnh, nhưng chưa đọc ra biển số."
            send_telegram_message(chat_id, msg)
            return jsonify({"ok": True}), 200

        elif message.get("text"):
            text = (message.get("text") or "").strip()

            if text == "/start":
                send_telegram_message(
                    chat_id,
                    "Gửi ảnh cà vẹt xe vào đây.\nBot sẽ crop từng cà vẹt, đọc biển số, lưu Drive và ghi Sheet."
                )
            else:
                send_telegram_message(chat_id, "📸 Hãy gửi ảnh cà vẹt xe.")
            return jsonify({"ok": True}), 200

        else:
            send_telegram_message(chat_id, "⚠️ Nội dung chưa hỗ trợ. Vui lòng gửi ảnh.")
            return jsonify({"ok": True}), 200

    except Exception as e:
        logger.exception("Webhook processing error")
        if chat_id:
            send_telegram_message(chat_id, f"❌ Lỗi xử lý: {str(e)}")
        return jsonify({"ok": False, "error": str(e)}), 200


# =========================
# START
# =========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
