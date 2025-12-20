import sys
from datetime import datetime, timedelta

import cv2
import peewee
import torch
from flask import Flask, Response, request
from flask_cors import CORS

from AsyncTaskQueue import AsyncTaskQueue
from T2I import generate
from camera_init import init_camera, release_camera, read_frame
from face_alignment.face_alignment import FaceAlignment
from face_detector.face_detector import DnnDetector
from model.model import Mini_Xception
from utils import histogram_equalization
from torchvision import transforms
import threading
import random

from db_models import Bottle
from vocabularies import ADJECTIVES, NOUNS, get_by_hex

app = Flask(__name__)
CORS(app, origins=["*"])

EMOJI_FOLDER = r'emoji'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
PHOTOS_FOLDER = r'photos'

# 全局变量
emojis = {}
cap = None
cap_lock = threading.Lock()
pretrained = 'checkpoint/model_weights/weights_epoch_75.pth.tar'
face_detector = None
face_alignment = None
device = None
mini_xception = None
task_queue = AsyncTaskQueue()

def clear_expired_bottle():
    twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
    Bottle.delete().where(Bottle.created_at <= twenty_four_hours_ago).execute()
def init_resources():
    """初始化所有资源"""
    global cap, face_detector, face_alignment, device, mini_xception

    sys.path.insert(1, 'face_detector')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(pretrained, map_location=device)
    mini_xception = Mini_Xception().to(device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    mini_xception.eval()

    face_alignment = FaceAlignment()
    face_detector = DnnDetector('face_detector')

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

        (x, y, w, h) = max(faces, key=lambda x: x[2] * x[3])
        input_face = face_alignment.frontalize_face((x, y, w, h), frame)
        input_face = cv2.resize(input_face, (48, 48))
        input_face = histogram_equalization(input_face)
        input_face = transform(input_face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = mini_xception(input_face)
            softmax = torch.nn.Softmax(dim=1)(outputs)
        return dict(zip(EMOTION_LABELS, softmax.flatten().tolist()))


    except Exception as e:
        return Response(f"Error in emotion detection: {e}", status=500)

@app.route('/vocabularies')
def vocabularies():
    return {
        "adjectives": ADJECTIVES,
        "nouns": NOUNS
    }

@app.route('/random_vocabulary')
def random_vocabulary():
    res = f"{random.randint(0, 255):02X}"
    while Bottle.select().where(Bottle.id == res).exists():
        res = f"{random.randint(0, 255):02X}"
    return {
        "id": res,
        "vocabulary": get_by_hex(res)
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



async def add_bottle(emotion: str, feeling: str, passage: str, hex_id: str):
    print(f"Generating {hex_id}")
    clear_expired_bottle()
    bottle = Bottle.create(
        id=hex_id,
        emotion=emotion,
        feeling=feeling,
        passage=passage,
        img_url=await generate(f"人物感受:{feeling}, 描述:{passage}生成没有性别特征的简笔画手绘风人物涂鸦")
    )
    bottle.save()
    print(f"saved bottle: {hex_id}")

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
    if emotion is None:
        return "Bad request", 400
    if feeling is None:
        return "Bad request", 400
    if passage is None:
        return "Bad request", 400
    if hex_id is None:
        return "Bad request", 400
    try:
        if int(hex_id, 16) > 255 or int(hex_id, 16) < 0:
            return "Bad request", 400
    except ValueError:
        return "Bad request", 400
    task_queue.add_task(add_bottle(emotion, feeling, passage, hex_id))
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
