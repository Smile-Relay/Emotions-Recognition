import sys
import os
import time

import cv2
import numpy as np
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
import base64

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
theme = "Apple"

def init_resources():
    """初始化所有资源"""
    global emojis, cap, face_detector, face_alignment, device, mini_xception

    sys.path.insert(1, 'face_detector')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(pretrained, map_location=device)
    mini_xception = Mini_Xception().to(device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    mini_xception.eval()

    face_alignment = FaceAlignment()
    face_detector = DnnDetector('face_detector')

    if not os.path.exists(PHOTOS_FOLDER):
        os.makedirs(PHOTOS_FOLDER)

    # 加载emoji图片
    emojis.clear()
    if os.path.exists(EMOJI_FOLDER):
        for t in os.listdir(EMOJI_FOLDER):
            theme_path = os.path.join(EMOJI_FOLDER, t)
            if not os.path.isdir(theme_path):
                continue
            emojis[t] = []
            for emotion in EMOTION_LABELS:
                emoji_path = os.path.join(theme_path, f"{emotion}.png")
                if os.path.exists(emoji_path):
                    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                    if emoji is not None:
                        emojis[t].append(emoji)
                    else:
                        emojis[t].append(np.zeros((48, 48, 4), dtype=np.uint8))
                else:
                    emojis[t].append(np.zeros((48, 48, 4), dtype=np.uint8))

    # 初始化摄像头
    global cap
    cap = init_camera()


@app.route('/set_theme')
def set_theme():
    global theme
    new_theme = request.args.get('theme')
    if new_theme is None:
        return Response("Bad request", status=400)
    if new_theme not in emojis:
        return Response("Theme not found", status=404)
    theme = new_theme
    return Response("OK")

@app.route('/get_theme')
def get_theme():
    global theme
    return Response(theme)


def generate_frames():
    """生成视频流帧"""
    global cap, theme

    if cap is None:
        init_resources()

    softmax = torch.nn.Softmax(dim=1)
    transform = transforms.ToTensor()

    while True:
        try:
            with cap_lock:
                ret, frame = read_frame()
            if not ret:
                with cap_lock:
                    release_camera()
                    cap = init_camera()
                continue

            faces = face_detector.detect_faces(frame)

            for (x, y, w, h) in faces:
                input_face = face_alignment.frontalize_face((x, y, w, h), frame)
                input_face = cv2.resize(input_face, (48, 48))
                input_face = histogram_equalization(input_face)
                input_face = transform(input_face).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = mini_xception(input_face)
                    emotion_idx = torch.argmax(outputs, dim=1).item()

                if theme in emojis and len(emojis[theme]) > emotion_idx:
                    emoji = emojis[theme][emotion_idx]
                    if emoji is not None and emoji.size > 0:
                        emoji_resized = cv2.resize(emoji, (w, h))
                        if emoji_resized.shape[2] == 4:
                            alpha_emoji = emoji_resized[:, :, 3] / 255.0
                            alpha_frame = 1.0 - alpha_emoji
                            for c in range(3):
                                frame[y:y+h, x:x+w, c] = (emoji_resized[:, :, c] * alpha_emoji +
                                                          frame[y:y+h, x:x+w, c] * alpha_frame)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"Error in frame generation: {e}")
            continue


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_photo', methods=['POST'])
def take_photo():
    photo_base64 = request.get_json().get("image")
    if photo_base64 is None:
        return Response("Bad request", status=400)
    if photo_base64.startswith("data:image"):
        photo_base64 = photo_base64.split(",")[1]

    decoded_bytes = base64.b64decode(photo_base64)
    with open(os.path.join(PHOTOS_FOLDER, f"{time.time()}.png"), "wb") as f:
        f.write(decoded_bytes)
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
