import os
import io
import cv2
import json
import numpy as np
import requests
import pytesseract
import re
import base64

from datetime import datetime
from flask import Flask, request, jsonify


TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")

TESS_LANG = "vie+eng"

app = Flask(__name__)


def tg(method):
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


def send(chat,text):
    requests.post(
        tg("sendMessage"),
        json={"chat_id":chat,"text":text}
    )


def get_file(file_id):

    r = requests.get(
        tg("getFile"),
        params={"file_id":file_id}
    )

    path = r.json()["result"]["file_path"]

    return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}"


def bytes_to_img(b):

    arr=np.frombuffer(b,np.uint8)

    return cv2.imdecode(arr,cv2.IMREAD_COLOR)


def img_to_bytes(img):

    _,buf=cv2.imencode(".jpg",img)

    return buf.tobytes()


def detect(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur=cv2.GaussianBlur(gray,(5,5),0)

    edges=cv2.Canny(blur,50,150)

    cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    res=[]

    for c in cnts:

        area=cv2.contourArea(c)

        if area<50000:
            continue

        x,y,w,h=cv2.boundingRect(c)

        ratio=w/float(h)

        if 1.2<ratio<2.5:

            res.append(img[y:y+h,x:x+w])

    if not res:
        return [img]

    return res[:20]


def preprocess(img):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )


def ocr(img):

    return pytesseract.image_to_string(
        img,
        lang=TESS_LANG,
        config="--psm 6"
    )


def extract_plate(text):

    raw=text.upper()

    raw=raw.replace("O","0")

    matches=re.findall(r"\d{2}[A-Z]\s?-?\d{3}[.\s]?\d{2}",raw)

    plates=[]

    for p in matches:

        p=re.sub(r"\s","",p)

        p=p.replace("-","")

        if len(p)==7:

            p=p[:3]+"-"+p[3:6]+"."+p[6:]

        plates.append(p)

    return list(set(plates))


def send_to_apps_script(img_bytes,plate,caption,chat,name):

    b64=base64.b64encode(img_bytes).decode()

    requests.post(
        APPS_SCRIPT_URL,
        json={
            "image":b64,
            "plate":plate,
            "caption":caption,
            "chatId":chat,
            "name":name
        }
    )


def process(file_id,chat,name,caption):

    url=get_file(file_id)

    img_bytes=requests.get(url).content

    img=bytes_to_img(img_bytes)

    crops=detect(img)

    found=[]

    for c in crops:

        pimg=preprocess(c)

        text=ocr(pimg)

        plates=extract_plate(text)

        img_bytes=img_to_bytes(c)

        if not plates:

            send_to_apps_script(
                img_bytes,
                "",
                caption,
                chat,
                name
            )

        else:

            for p in plates:

                found.append(p)

                send_to_apps_script(
                    img_bytes,
                    p,
                    caption,
                    chat,
                    name
                )

    return found


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

        else:

            send(chat,"📸 Gửi ảnh cà vẹt để quét")

            return "ok"

        if plates:

            send(chat,"✅ Đã quét:\n"+ "\n".join(plates))

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
