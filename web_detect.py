import asyncio
import sys
from datetime import datetime, timedelta

import cv2
import peewee
import torch
from flask import Flask, Response, request
from flask_cors import CORS

from camera_init import init_camera, release_camera, read_frame
from face_alignment.face_alignment import FaceAlignment
from face_detector.face_detector import DnnDetector
from model.model import Mini_Xception
from utils import histogram_equalization
from torchvision import transforms
import threading
import random
from pyppeteer import launch
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
import cups

from db_models import Bottle

app = Flask(__name__)
CORS(app, origins=["*"])

EMOJI_FOLDER = r'emoji'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
PHOTOS_FOLDER = r'photos'
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
print_options = {
    "media": "4x6",
    "fit-to-page": "True"
}

# 全局变量
emojis = {}
cap = None
cap_lock = threading.Lock()
pretrained = 'checkpoint/model_weights/weights_epoch_75.pth.tar'
face_detector = None
face_alignment = None
device = None
mini_xception = None
gender_model = None
executor = ThreadPoolExecutor(max_workers=2)

async def take_screenshot(url, selector):
    browser = await launch(
        args=[
            "--disable-infobars",
            "--disable-component-update",
            "--password-store=basic",
            "--headless=new",
            "--disable-gpu"
        ],
        headless=True,
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False,
        dumpio=True,
        executablePath='/usr/bin/chromium'
    )
    print("Chromium started")
    page = await browser.newPage()
    await page.setViewport({'width': 800, 'height': 1280})
    await page.goto(url)
    print(1)
    await page.waitForSelector(selector, {'visible': True})
    print(2)
    await page.waitForFunction("""
      () => {
        const el = document.getElementById('render-complete');
        return el && el.getAttribute('data-ready') === 'true';
      }
    """)
    print(3)
    await page.evaluate(f"""
            async () => {{
                const imgs = Array.from(document.querySelectorAll('{selector} img'));
                await Promise.all(imgs.map(img => {{
                    if (img.complete) return;
                    return new Promise(resolve => img.onload = resolve);
                }}));
            }}
        """)
    print(4)
    element = await page.querySelector('#bottle')
    img_bytes = await element.screenshot()
    await browser.close()
    conn = cups.Connection()
    printers = conn.getPrinters()

    default_printer = conn.getDefault()
    if not default_printer:
        default_printer = list(printers.keys())[0]

    with NamedTemporaryFile(suffix=".png") as f:
        f.write(img_bytes)
        f.flush()
        print("printing " + f.name)
        conn.printFile(default_printer, f.name, "Bottle Print", print_options)

def run_async_task(url, selector):
    asyncio.run(take_screenshot(url, selector))


def clear_expired_bottle():
    twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
    Bottle.delete().where(Bottle.created_at <= twenty_four_hours_ago).execute()

def init_resources():
    """初始化所有资源"""
    global cap, face_detector, face_alignment, device, mini_xception, gender_model
    sys.path.insert(1, 'face_detector')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(pretrained, map_location=device)
    mini_xception = Mini_Xception().to(device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    mini_xception.eval()

    face_alignment = FaceAlignment()
    face_detector = DnnDetector('face_detector')

    gender_model = cv2.dnn.readNet(genderModel,genderProto)

    # 初始化摄像头
    global cap
    cap = init_camera()


@app.route('/detect')
def detect():
    global cap
    if cap is None:
        init_resources()
    transform = transforms.ToTensor()
    try:
        for _ in range(5):
            with cap_lock:
                ret, frame = read_frame()
            if ret:
                break
            with cap_lock:
                release_camera()
                cap = init_camera()
        else:
            return Response("Error in camera init", status=500)

        faces = face_detector.detect_faces(frame)
        if not faces:
            return Response("No face detected", status=404)
        faceBox = max(faces, key=lambda x: x[2] * x[3])
        input_face = face_alignment.frontalize_face(faceBox, frame)
        blob = cv2.dnn.blobFromImage(input_face, 1.0, (227, 227), MODEL_MEAN_VALUES)
        input_face = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
        input_face = cv2.resize(input_face, (48, 48))
        input_face = histogram_equalization(input_face)
        input_face = transform(input_face).unsqueeze(0).to(device)

        gender_model.setInput(blob)
        genderPreds = gender_model.forward()
        gender = genderPreds[0].argmax()

        with torch.no_grad():
            outputs = mini_xception(input_face)
            softmax = torch.nn.Softmax(dim=1)(outputs)
        prediction = dict(zip(EMOTION_LABELS, softmax.flatten().tolist()))
        prediction['Gender'] = int(gender)
        print(prediction)
        return prediction
    except Exception as e:
        return Response(f"Error in emotion detection: {e}", status=500)

@app.route('/random_vocabulary')
def random_vocabulary():
    res = f"{random.randint(0, 255):02X}"
    while Bottle.select().where(Bottle.id == res).exists():
        res = f"{random.randint(0, 255):02X}"
    return {
        "id": res
    }

@app.route('/random_id')
def random_id():
    clear_expired_bottle()
    query = Bottle.select().order_by(peewee.fn.Random()).limit(1)
    item = query.first()
    if item is None:
        return "There are not any bottle", 404
    return item.id

@app.route('/get_bottle/<string:id>')
def get_bottle(id: str):
    clear_expired_bottle()
    query = Bottle.select().where(Bottle.id == id)
    item = query.first()
    if item is None:
        return "Bottle not found", 404
    return item.__data__

@app.route('/comment/<string:id>')
def comment(id: str):
    clear_expired_bottle()
    query = Bottle.select().where(Bottle.id == id)
    item = query.first()
    comment_type = request.args.get('type')
    if comment_type not in ["like", "hug", "flower"]:
        return "Bad request", 400
    if not item:
        return "Bottle not found", 404
    if comment_type == "like":
        item.likes += 1
    if comment_type == "hug":
        item.hugs += 1
    if comment_type == "flower":
        item.flowers += 1
    item.save()
    return "OK"


@app.route("/throw", methods=['POST'])
def throw():
    emotion = request.get_json().get("emotion")
    feeling = request.get_json().get("feeling")
    passage = request.get_json().get("passage")
    hex_id = request.get_json().get("id")
    img_url = request.get_json().get("img_url")
    lang = request.get_json().get("lang")
    if emotion is None:
        return "Bad request", 400
    if feeling is None:
        return "Bad request", 400
    if passage is None:
        return "Bad request", 400
    if hex_id is None:
        return "Bad request", 400
    if img_url is None:
        return "Bad request", 400
    if lang is None:
        return "Bad request", 400
    try:
        if int(hex_id, 16) > 255 or int(hex_id, 16) < 0:
            return "Bad request", 400
    except ValueError:
        return "Bad request", 400
    clear_expired_bottle()
    bottle = Bottle.create(
        id=hex_id,
        emotion=emotion,
        feeling=feeling,
        passage=passage,
        img_url=img_url
    )
    bottle.save()
    executor.submit(run_async_task, f"http://localhost:3000/{lang}/view?id={hex_id}", "#bottle")
    return "OK"

@app.teardown_appcontext
def cleanup(exception):
    global cap
    with cap_lock:
        if cap is not None:
            release_camera()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        init_resources()
        app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
    except Exception as e:
        print(f"Error starting application: {e}")
    finally:
        with cap_lock:
            if cap is not None:
                release_camera()
        cv2.destroyAllWindows()
