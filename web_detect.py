import sys
import cv2
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

@app.route("/throw", methods=['POST'])
def throw():
    pass

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
