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
size_faceROI = 256

st.title("Streamlit App: Mouth shape recognition")
st.write("Kyutech, Saitoh-lab")
st.markdown("---")

target_person_id = st.selectbox("Target person", ("P001", "P002", "P003"))
st.write("You selected:", target_person_id)
st.markdown("---")


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
