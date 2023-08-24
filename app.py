import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np
import cv2
import math
from PIL import Image
import mediapipe as mp
import subprocess
import torch
from torchvision import transforms

# left eye contour
landmark_left_eye_points = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]
# right eye contour
landmark_right_eye_points = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

size_LFROI = 200
size_faceROI =  256

st.title("Streamlit App: Mouth shape recognition")
st.write("Kyutech, Saitoh-lab")
st.markdown("---")

target_person_id = st.selectbox("Target person", ("P001", "P002", "P003"))
st.write("You selected:", target_person_id)
st.markdown("---")


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        # モノクロ
        pass
    elif new_image.shape[2] == 3:
        # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
        
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:
        # モノクロ
        pass
    elif new_image.shape[2] == 3:
        # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]

    new_image = Image.fromarray(new_image)

    return new_image


def func(value1, value2):
    return int(value1 * value2)


def LFROI_extraction_sub(image, face_points0):
    image_height, image_width, channels = image.shape[:3]

    image_cx = image_width / 2
    image_cy = image_height / 2

    left_eye_x = face_points0[33][0]
    left_eye_y = face_points0[33][1]
    left_eye_point = (left_eye_x, left_eye_y)

    right_eye_x = face_points0[263][0]
    right_eye_y = face_points0[263][1]
    right_eye_point = (right_eye_x, right_eye_y)

    nose_x = face_points0[2][0]
    nose_y = face_points0[2][1]
    nose_point = (nose_x, nose_y)

    eye_distance2 = (left_eye_x - right_eye_x) * (left_eye_x - right_eye_x) + (left_eye_y - right_eye_y) * (left_eye_y - right_eye_y)
    eye_distance = math.sqrt(eye_distance2)

    eye_angle = math.atan((left_eye_y - right_eye_y) / (left_eye_x - right_eye_x))

    scale = 200.0 / eye_distance
    cx = nose_x
    cy = nose_y

    mat_rot = cv2.getRotationMatrix2D((cx, cy), eye_angle, scale)
    tx = image_cx - cx
    ty = image_cy - cy
    mat_tra = np.float32([[1, 0, tx], [0, 1, ty]])

    image_width_ = int(image_width)
    image_height_ = int(image_height)

    normalized_image1 = cv2.warpAffine(image, mat_rot, (image_width_, image_height_))
    normalized_image2 = cv2.warpAffine(normalized_image1, mat_tra, (image_width_, image_height_))

    face_points1 = []
    for p0 in face_points0:
        x0 = p0[0]
        y0 = p0[1]
        x1 = mat_rot[0][0] * x0 + mat_rot[1][0] * y0 + mat_rot[0][2]
        y1 = mat_rot[0][1] * x0 + mat_rot[1][1] * y0 + mat_rot[1][2]
        x2 = x1 + mat_tra[0][2]
        y2 = y1 + mat_tra[1][2]

        face_points1.append((x2, y2))

    cx = face_points1[2][0]
    left = int(cx - size_LFROI / 2)
    top = int(face_points1[2][1] - size_LFROI / 4)
    right = left + size_LFROI
    bottom = top + size_LFROI

    return (left, top, right, bottom), normalized_image2, face_points1


def LFROI_extraction(image):
    out_image = image.copy()

    black_image = np.zeros((size_LFROI, size_LFROI, 3), np.uint8)
    white_image = black_image + 200

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image)
        (image_height, image_width) = image.shape[:2]
      
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                points = []
                for landmark in face.landmark:
                    x = func(landmark.x, image_width)
                    y = func(landmark.y, image_height)
                    cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                    cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                    points.append((x, y))

                rect_LFROI, normalized_image_LFROI, new_points_LFROI = LFROI_extraction_sub(image, points)
                LFROI = normalized_image_LFROI[rect_LFROI[1]: rect_LFROI[3], rect_LFROI[0]: rect_LFROI[2]]

            return LFROI

    return white_image


def preprocess(image, transform):   
    image = transform(image)  # PIL
    C, H, W = image.shape
    image = image.reshape(1, C, H, W)
    
    return image


def make_graph_image(values):
    graph_image = np.zeros((120, 120, 3), np.uint8)

    fontface = cv2.FONT_HERSHEY_PLAIN
    label = ["N", "a", "i", "u", "e", "o"]
    fontscale = 1.0
    thickness = 1

    max_idx = np.argmax(values)

    x0 = 20
    for idx, v in enumerate(values):
        x1 = x0 + int(v * 100)
        y0 = idx * 20
        y1 = (idx + 1) * 20
        if idx == max_idx:
            cv2.rectangle(graph_image, (x0, y0), (x1, y1), (0, 0, 255), -1)
            cv2.rectangle(graph_image, (x0+1, y0+1), (x1-1, y1-1), (200, 200, 255), -1)
        else:
            cv2.rectangle(graph_image, (x0, y0), (x1, y1), (0, 255, 0), -1)
            cv2.rectangle(graph_image, (x0+1, y0+1), (x1-1, y1-1), (200, 255, 200), -1)

        (w, h), baseline = cv2.getTextSize(label[idx], fontface, fontscale, thickness)
        x = int((x0 - w) / 2)
        cv2.putText(graph_image, label[idx], (x, y1-3), fontface, fontscale, (255, 255, 255), thickness)
        
    return graph_image


def prediction(model, crop_image):
    # モデルを評価モードにする
    model.eval()

    with torch.no_grad():
        # 予測
        outputs = model(crop_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        #st.write(probabilities)

        # obtain first six classes
        probabilities6 = probabilities[0:6]
        #st.write(probabilities6)

        graph_image = make_graph_image(probabilities6)
        
        # 予測結果をクラス番号に変換
        _, predicted = torch.max(outputs, 1)

    return predicted, graph_image

def faceROI_extraction(image, face_points0):
    image_height, image_width, channels = image.shape[:3]

    image_cx = image_width / 2
    image_cy = image_height / 2

    left_eye_x = 0
    left_eye_y = 0
    for idx in landmark_left_eye_points:
        x = face_points0[idx][0]
        y = face_points0[idx][1]
        left_eye_x += x
        left_eye_y += y

    left_eye_x = int(left_eye_x / len(landmark_left_eye_points))
    left_eye_y = int(left_eye_y / len(landmark_left_eye_points))

    right_eye_x = 0
    right_eye_y = 0
    for idx in landmark_right_eye_points:
        x = face_points0[idx][0]
        y = face_points0[idx][1]
        right_eye_x += x
        right_eye_y += y

    right_eye_x = int(right_eye_x / len(landmark_right_eye_points))
    right_eye_y = int(right_eye_y / len(landmark_right_eye_points))


    eye_distance2 = (left_eye_x - right_eye_x) * (left_eye_x - right_eye_x) + (left_eye_y - right_eye_y) * (left_eye_y - right_eye_y)
    eye_distance = math.sqrt(eye_distance2)
    
    eye_angle = math.atan((left_eye_y - right_eye_y) / (left_eye_x - right_eye_x))

    target_eye_distance = 55

    scale = target_eye_distance / eye_distance

    cx = (left_eye_x + right_eye_x) / 2
    cy = (left_eye_y + right_eye_y) / 2

    mat_rot = cv2.getRotationMatrix2D((cx, cy), eye_angle, scale)
    tx = image_cx - cx
    ty = image_cy - cy
    mat_tra = np.float32([[1, 0, tx], [0, 1, ty]])

    image_width_ = int(image_width)
    image_height_ = int(image_height)

    normalized_image1 = cv2.warpAffine(image, mat_rot, (image_width_, image_height_))
    normalized_image2 = cv2.warpAffine(normalized_image1, mat_tra, (image_width_, image_height_))

    face_points1 = []
    for p0 in face_points0:
        x0 = p0[0]
        y0 = p0[1]
        x1 = mat_rot[0][0] * x0 + mat_rot[1][0] * y0 + mat_rot[0][2]
        y1 = mat_rot[0][1] * x0 + mat_rot[1][1] * y0 + mat_rot[1][2]
        x2 = x1 + mat_tra[0][2]
        y2 = y1 + mat_tra[1][2]

        face_points1.append((x2, y2))

    left = int(image_cx - size_faceROI / 2)
    top = int(image_cy -size_faceROI / 2 + 33)
    right = left + size_faceROI
    bottom = top + size_faceROI

    return (left, top, right, bottom), normalized_image2, face_points1


def process(image, is_show_image, draw_pattern):
    out_image = image.copy()

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image)
        (image_height, image_width) = image.shape[:2]
      
        black_image = np.zeros((image_height, image_width, 3), np.uint8)
        white_image = black_image + 200

        if is_show_image == False:
            out_image = white_image.copy()

        if draw_pattern == "A":
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    for landmark in face.landmark:
                        x = func(landmark.x, image_width)
                        y = func(landmark.y, image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)

        elif draw_pattern == "B":
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    points = []
                    for landmark in face.landmark:
                        x = func(landmark.x, image_width)
                        y = func(landmark.y, image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                        points.append((x, y))

                    rect_faceROI, normalized_image_faceROI, new_points_faceROI = faceROI_extraction(image, points)
                    faceROI = normalized_image_faceROI[rect_faceROI[1]: rect_faceROI[3], rect_faceROI[0]: rect_faceROI[2]]
                    faceROI = cv2.resize(faceROI, (100, 100))

                    out_image[0: 100, 0: 100] = faceROI

    return cv2.flip(out_image, 1)


# data transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# vowel dict
idxtovowel = {0: '閉口', 1: 'あ', 2: 'い', 3: 'う', 4: 'え', 5: 'お'}
# training model path
model_path = 'model/model_mobilenetv2.pth'
# load model
model = torch.load(model_path)
# load device : cpu
device = torch.device("cpu")
model.to(device)

magrin = 5

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def __init__(self) -> None:
        self.is_show_image = True
        self.draw_pattern = "A"

    def recv(self, frame):
        image_cv = frame.to_ndarray(format="bgr24")

        image_cv = process(image_cv, self.is_show_image, self.draw_pattern)
        
        return av.VideoFrame.from_ndarray(image_cv, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)


if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.is_show_image = st.checkbox("show camera image", value=True)
    webrtc_ctx.video_processor.draw_pattern = st.radio("select draw pattern", ["A", "B", "None"], key="A", horizontal=True)
