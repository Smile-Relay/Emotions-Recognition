import sys

import cv2
import numpy as np
import os

import torch
from flask import Flask, Response
from camera_init import init_camera, release_camera, read_frame
from face_alignment.face_alignment import FaceAlignment
from face_detector.face_detector import DnnDetector
from model.model import Mini_Xception
from utils import histogram_equalization, get_label_emotion
import torchvision.transforms.transforms as transforms

app = Flask(__name__)

EMOJI_FOLDER = r'emoji'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 初始化变量
emojis = []
cap = None
pretrained = 'checkpoint/model_weights/weights_epoch_75.pth.tar'
face_detector = None
face_alignment = None
device = None
mini_xception = None


def init_resources():
    """初始化所有资源"""
    global emojis, cap, face_detector, face_alignment, device, mini_xception

    sys.path.insert(1, 'face_detector')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pretrained, map_location=device)
    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    face_alignment = FaceAlignment()

    face_detector = DnnDetector('face_detector')

    # 加载emoji图片
    emojis = []
    if os.path.exists(EMOJI_FOLDER):
        for emotion in EMOTION_LABELS:
            emoji_path = os.path.join(EMOJI_FOLDER, f"{emotion}.png")
            if os.path.exists(emoji_path):
                emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                if emoji is not None:
                    emojis.append(emoji)
                else:
                    # 使用默认表情
                    emojis.append(np.zeros((48, 48, 4), dtype=np.uint8))
            else:
                emojis.append(np.zeros((48, 48, 4), dtype=np.uint8))
    else:
        # 如果emoji文件夹不存在，创建空的emoji数组
        emojis = [np.zeros((48, 48, 4), dtype=np.uint8) for _ in EMOTION_LABELS]

    # 初始化摄像头
    cap = init_camera()


def generate_frames():
    """生成视频流帧"""
    global cap

    # 确保资源已初始化
    if cap is None:
        init_resources()

    while True:
        try:
            # 读取帧
            ret, frame = read_frame()
            if not ret:
                # 如果读取失败，尝试重新打开摄像头
                release_camera()
                init_camera()
                continue

            faces = face_detector.detect_faces(frame)


            for (x, y, w, h) in faces:

                input_face = face_alignment.frontalize_face((x, y, w, h), frame)
                input_face = cv2.resize(input_face, (48, 48))

                input_face = histogram_equalization(input_face)

                input_face = transforms.ToTensor()(input_face).to(device)
                input_face = torch.unsqueeze(input_face, 0)

                with torch.no_grad():
                    input_face = input_face.to(device)
                    emotion = mini_xception(input_face)
                    # print(f'\ntime={(time.time()-t) * 1000 } ms')

                    torch.set_printoptions(precision=6)
                    softmax = torch.nn.Softmax()
                    emotions_soft = softmax(emotion.squeeze()).reshape(-1, 1).cpu().detach().numpy()
                    emotions_soft = np.round(emotions_soft, 3)

                    emotion = torch.argmax(emotion)
                    emotion_idx = emotion.squeeze().cpu().detach().item()

                    if emojis[emotion_idx] is not None and emojis[emotion_idx].size > 0:
                        emoji = cv2.resize(emojis[emotion_idx], (w, h))

                        # 确保emoji有alpha通道
                        if emoji.shape[2] == 4:
                            # 叠加emoji（带透明度）
                            alpha_emoji = emoji[:, :, 3] / 255.0
                            alpha_frame = 1.0 - alpha_emoji

                            for c in range(0, 3):
                                frame[y:y + h, x:x + w, c] = (emoji[:, :, c] * alpha_emoji +
                                                              frame[y:y + h, x:x + w, c] * alpha_frame)


            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            # 捕获所有异常，确保视频流不中断
            print(f"Error in frame generation: {e}")
            continue


@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.teardown_appcontext
def cleanup(exception):
    """应用关闭时清理资源"""
    global cap
    if cap is not None:
        release_camera()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        # 初始化资源
        init_resources()

        # 启动Flask应用
        app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)

    except Exception as e:
        print(f"Error starting application: {e}")
    finally:
        # 确保资源释放
        if cap is not None:
            release_camera()
        cv2.destroyAllWindows()