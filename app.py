import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
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


st.title("Streamlit App: Face motion by MediaPipe")
st.write("Kyutech, Saitoh-lab")

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


def process(image, is_show_image, draw_pattern):
    out_image = image.copy()

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        # landmark indexes
        all_left_eye_idxs = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
        all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))
        all_right_eye_idxs = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
        all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
        all_left_brow_idxs = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)
        all_left_brow_idxs = set(np.ravel(all_left_brow_idxs))
        all_right_brow_idxs = list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)
        all_right_brow_idxs = set(np.ravel(all_right_brow_idxs))
        all_lip_idxs = list(mp.solutions.face_mesh.FACEMESH_LIPS)
        all_lip_idxs = set(np.ravel(all_lip_idxs))
        all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
        all_idxs = all_idxs.union(all_left_brow_idxs)
        all_idxs = all_idxs.union(all_right_brow_idxs)
        all_idxs = all_idxs.union(all_lip_idxs)

        left_iris_idxs = list(mp.solutions.face_mesh.FACEMESH_LEFT_IRIS)
        left_iris_idxs = set(np.ravel(left_iris_idxs))
        right_iris_idxs = list(mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS)
        right_iris_idxs = set(np.ravel(right_iris_idxs))

        results = face_mesh.process(image)

        (image_height, image_width) = image.shape[:2]

        black_image = np.zeros((image_height, image_width, 3), np.uint8)
        white_image = black_image + 200

        if is_show_image == False:
            out_image = white_image.copy()

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
               for landmark in face.landmark:               
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                    cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)

    return cv2.flip(out_image, 1)


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
    webrtc_ctx.video_processor.draw_pattern = st.radio("select draw pattern", ["A", "B", "C", "None"], key="A", horizontal=True)
