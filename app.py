import os
import cv2
import json
import numpy as np
import requests
import pytesseract
import re
import base64
from flask import Flask, request, jsonify

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")
TESS_LANG = os.getenv("TESS_LANG", "vie+eng")

app = Flask(__name__)


def tg(method):
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


def send(chat_id, text):
    requests.post(
        tg("sendMessage"),
        json={"chat_id": chat_id, "text": text},
        timeout=60
    )


def get_file(file_id):
    r = requests.get(
        tg("getFile"),
        params={"file_id": file_id},
        timeout=60
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise Exception(f"Telegram getFile lỗi: {data}")
    path = data["result"]["file_path"]
    return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}"


def bytes_to_img(b):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception("Không decode được ảnh")
    return img


def img_to_bytes(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise Exception("Không encode được ảnh JPG")
    return buf.tobytes()


def detect_cards(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50000:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)

        # cà vẹt ngang
        if 1.2 < ratio < 2.5 and w > 300 and h > 150:
            res.append((x, y, img[y:y+h, x:x+w]))

    if not res:
        return [img]

    # sort trái -> phải, trên -> dưới để ổn định
    res.sort(key=lambda t: (t[1], t[0]))
    return [x[2] for x in res[:20]]


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    return th


def normalize_plate_digits(plate_raw):
    """
    Chuyển:
    50F-054.23
    50F054.23
    50F 054 23
    -> 50F05423
    """
    p = plate_raw.upper()
    p = p.replace("O", "0")
    p = re.sub(r"[^0-9A-Z]", "", p)
    return p


def find_best_plate(text):
    """
    Chỉ nhận đúng format 2 số + 1 chữ + 5 số
    Ví dụ: 50F05423
    """
    if not text:
        return None

    raw = text.upper()
    raw = raw.replace("O", "0")
    raw = raw.replace("I", "1")
    raw = raw.replace("L", "1")
    raw = raw.replace(",", ".")
    raw = raw.replace("–", "-").replace("—", "-")

    # Tìm các dạng phổ biến
    patterns = [
        r"\b\d{2}[A-Z]\s?-?\s?\d{3}[.\s]?\d{2}\b",
        r"\b\d{2}[A-Z]\d{5}\b"
    ]

    candidates = []
    for p in patterns:
        candidates.extend(re.findall(p, raw))

    normalized = []
    for c in candidates:
        n = normalize_plate_digits(c)
        # chỉ lấy đúng 8 ký tự: 2 số + 1 chữ + 5 số
        if re.fullmatch(r"\d{2}[A-Z]\d{5}", n):
            normalized.append(n)

    if not normalized:
        return None

    # ưu tiên biển số bắt đầu bằng 2 số tỉnh/thành
    # lấy candidate đầu tiên
    return normalized[0]


def extract_plate_from_crop(card_img):
    """
    Chỉ OCR vùng ưu tiên là phần dưới bên trái của cà vẹt,
    để tránh đọc nhầm số khác như 22A22084.
    """
    h, w = card_img.shape[:2]

    rois = []

    # ROI 1: nửa dưới bên trái
    roi1 = card_img[int(h * 0.45):int(h * 0.92), 0:int(w * 0.62)]
    if roi1.size > 0:
        rois.append(roi1)

    # ROI 2: gần giữa dưới
    roi2 = card_img[int(h * 0.40):int(h * 0.88), int(w * 0.05):int(w * 0.75)]
    if roi2.size > 0:
        rois.append(roi2)

    # ROI 3: toàn card fallback
    rois.append(card_img)

    all_text = []

    for roi in rois:
        txt1 = pytesseract.image_to_string(roi, lang=TESS_LANG, config="--psm 6")
        all_text.append(txt1)

        pre = preprocess(roi)
        txt2 = pytesseract.image_to_string(pre, lang=TESS_LANG, config="--psm 6")
        all_text.append(txt2)

        combined = "\n".join(all_text)
        plate = find_best_plate(combined)
        if plate:
            return plate, combined

    return None, "\n".join(all_text)


def send_to_apps_script(img_bytes, plate, caption, chat, name, ocr_text):
    if not APPS_SCRIPT_URL:
        raise Exception("Thiếu APPS_SCRIPT_URL")

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "image": b64,
        "plate": plate or "",
        "caption": caption or "",
        "chatId": str(chat),
        "name": name or "",
        "ocrText": ocr_text or ""
    }

    r = requests.post(
        APPS_SCRIPT_URL,
        json=payload,
        timeout=120
    )

    # Bắt lỗi rõ
    r.raise_for_status()

    try:
        data = r.json()
    except Exception:
        raise Exception(f"Apps Script không trả JSON hợp lệ: {r.text[:500]}")

    if data.get("status") != "ok":
        raise Exception(f"Apps Script lỗi: {data}")

    return data


def process(file_id, chat, name, caption):
    url = get_file(file_id)

    img_bytes = requests.get(url, timeout=120).content
    img = bytes_to_img(img_bytes)

    crops = detect_cards(img)

    found = []
    saved_count = 0

    for c in crops:
        plate, ocr_text = extract_plate_from_crop(c)
        crop_bytes = img_to_bytes(c)

        result = send_to_apps_script(
            crop_bytes,
            plate,
            caption,
            chat,
            name,
            ocr_text
        )

        saved_count += 1

        if plate:
            found.append(plate)

    return found, saved_count


@app.route("/telegram/webhook", methods=["POST"])
def webhook():
    data = request.json or {}
    msg = data.get("message")

    if not msg:
        return "ok"

    chat = msg["chat"]["id"]
    user = msg.get("from", {})
    name = f'{user.get("first_name","")} {user.get("last_name","")}'.strip()
    caption = msg.get("caption", "")

    try:
        if "photo" in msg:
            file_id = msg["photo"][-1]["file_id"]
            plates, saved_count = process(file_id, chat, name, caption)

        elif "document" in msg:
            file_id = msg["document"]["file_id"]
            plates, saved_count = process(file_id, chat, name, caption)

        else:
            send(chat, "📸 Gửi ảnh cà vẹt để quét")
            return "ok"

        if plates:
            unique_plates = list(dict.fromkeys(plates))
            send(
                chat,
                "✅ Đã quét và lưu thành công.\n"
                f"Số ảnh crop đã lưu: {saved_count}\n"
                "Biển số:\n" + "\n".join(unique_plates)
            )
        else:
            send(
                chat,
                "⚠️ Đã lưu ảnh crop nhưng chưa đọc chắc chắn ra biển số."
                f"\nSố ảnh crop đã lưu: {saved_count}"
            )

    except Exception as e:
        send(chat, f"❌ Lỗi xử lý: {str(e)}")

    return "ok"


@app.route("/")
def home():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
