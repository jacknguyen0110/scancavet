import os
import io
import cv2
import json
import numpy as np
import requests
import pytesseract
import re
from datetime import datetime
from flask import Flask, request, jsonify

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


# =========================
# ENV
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
SHEET_NAME = os.getenv("SHEET_NAME")

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

TESS_LANG = os.getenv("TESS_LANG", "vie+eng")

OWNER_EMAIL = "jacknguyen0110@gmail.com"

app = Flask(__name__)


# =========================
# GOOGLE AUTH
# =========================

def get_google_credentials():

    info = json.loads(SERVICE_ACCOUNT_JSON)

    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets"
    ]

    return Credentials.from_service_account_info(info, scopes=scopes)


def get_drive():

    creds = get_google_credentials()

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_sheet():

    gc = gspread.authorize(get_google_credentials())

    sh = gc.open_by_key(SPREADSHEET_ID)

    try:

        ws = sh.worksheet(SHEET_NAME)

    except:

        ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=20)

    return ws


# =========================
# TELEGRAM
# =========================

def tg(method):

    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def send(chat_id, text):

    requests.post(
        tg("sendMessage"),
        json={"chat_id": chat_id, "text": text}
    )


def get_file(file_id):

    r = requests.get(tg("getFile"), params={"file_id": file_id})

    path = r.json()["result"]["file_path"]

    return f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{path}"


# =========================
# DRIVE
# =========================

def upload(img_bytes, filename):

    drive = get_drive()

    media = MediaIoBaseUpload(
        io.BytesIO(img_bytes),
        mimetype="image/jpeg"
    )

    metadata = {
        "name": filename,
        "parents": [DRIVE_FOLDER_ID]
    }

    file = drive.files().create(
        body=metadata,
        media_body=media,
        fields="id,webViewLink"
    ).execute()

    file_id = file["id"]

    # share to your gmail
    try:

        drive.permissions().create(
            fileId=file_id,
            body={
                "type": "user",
                "role": "reader",
                "emailAddress": OWNER_EMAIL
            }
        ).execute()

    except:
        pass

    return file_id, file["webViewLink"]


# =========================
# IMAGE
# =========================

def bytes_to_img(b):

    arr = np.frombuffer(b, np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def img_to_bytes(img):

    _, buf = cv2.imencode(".jpg", img)

    return buf.tobytes()


# =========================
# DETECT CAVET
# =========================

def detect(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5),0)

    edges = cv2.Canny(blur,50,150)

    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    out=[]

    for c in cnts:

        area=cv2.contourArea(c)

        if area<50000:
            continue

        x,y,w,h=cv2.boundingRect(c)

        ratio=w/float(h)

        if 1.2<ratio<2.5:

            out.append(img[y:y+h,x:x+w])

    if not out:
        return [img]

    return out[:20]


# =========================
# OCR
# =========================

def preprocess(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,15
    )


def ocr(img):

    return pytesseract.image_to_string(
        img,
        lang=TESS_LANG,
        config="--psm 6"
    )


def plate(text):

    raw=text.upper()

    raw=raw.replace("O","0")

    m=re.findall(r"\d{2}[A-Z]\s?-?\d{3}[.\s]?\d{2}",raw)

    out=[]

    for p in m:

        p=re.sub(r"\s","",p)

        p=p.replace("-","")

        if len(p)==7:

            p=p[:3]+"-"+p[3:6]+"."+p[6:]

        out.append(p)

    return list(set(out))


# =========================
# PROCESS
# =========================

def process(file_id,chat,name,caption):

    url=get_file(file_id)

    img_bytes=requests.get(url).content

    img=bytes_to_img(img_bytes)

    crops=detect(img)

    sheet=get_sheet()

    found=[]

    for i,c in enumerate(crops):

        pimg=preprocess(c)

        text=ocr(pimg)

        plates=plate(text)

        b=img_to_bytes(c)

        fname=f"cavet_{datetime.now().timestamp()}_{i}.jpg"

        fid,link=upload(b,fname)

        thumb=f'=IMAGE("https://drive.google.com/thumbnail?id={fid}&sz=w300")'

        if not plates:

            sheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                chat,
                name,
                caption,
                "",
                link,
                thumb,
                text,
                "NO_PLATE"
            ])

        else:

            for p in plates:

                found.append(p)

                sheet.append_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    chat,
                    name,
                    caption,
                    p,
                    link,
                    thumb,
                    text,
                    "OK"
                ])

    return found


# =========================
# WEBHOOK
# =========================

@app.route("/telegram/webhook",methods=["POST"])
def webhook():

    data=request.json

    msg=data.get("message")

    if not msg:
        return "ok"

    chat=msg["chat"]["id"]

    user=msg.get("from",{})

    name=f'{user.get("first_name","")} {user.get("last_name","")}'

    caption=msg.get("caption","")

    try:

        if "photo" in msg:

            file_id=msg["photo"][-1]["file_id"]

            plates=process(file_id,chat,name,caption)

        elif "document" in msg:

            file_id=msg["document"]["file_id"]

            plates=process(file_id,chat,name,caption)

        elif "text" in msg:

            send(chat,"📸 Gửi ảnh cà vẹt để quét")

            return "ok"

        if plates:

            send(chat,"✅ Đã quét:\n"+"\n".join(plates))

        else:

            send(chat,"⚠️ Không đọc được biển số")

    except Exception as e:

        send(chat,f"❌ Lỗi xử lý: {str(e)}")

    return "ok"


@app.route("/")
def home():

    return jsonify({"status":"running"})


if __name__=="__main__":

    port=int(os.environ.get("PORT",8080))

    app.run(host="0.0.0.0",port=port)