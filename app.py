import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.title("Streamlit App: Face motion by MediaPipe")
st.write("Kyutech, Saitoh-lab")

def func(value1, value2):
    return int(value1 * value2)


def drawB(image, face, image_width, image_height):
    left_eye_idxs = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133]
    right_eye_idxs = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
    left_eyebrow_idxs = [55, 65, 52, 53, 46]
    right_eyebrow_idxs = [285, 295, 282, 283, 276]
    lip_idxs = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]

    for i in range(len(left_eye_idxs)-1):
        idx1 = left_eye_idxs[i]
        idx2 = left_eye_idxs[i+1]
        x1 = func(face.landmark[idx1].x, image_width)
        y1 = func(face.landmark[idx1].y, image_height)
        x2 = func(face.landmark[idx2].x, image_width)
        y2 = func(face.landmark[idx2].y, image_height)

        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
        
    for i in range(len(right_eye_idxs)-1):
        idx1 = right_eye_idxs[i]
        idx2 = right_eye_idxs[i+1]
        x1 = func(face.landmark[idx1].x, image_width)
        y1 = func(face.landmark[idx1].y, image_height)
        x2 = func(face.landmark[idx2].x, image_width)
        y2 = func(face.landmark[idx2].y, image_height)

        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

    for i in range(len(left_eyebrow_idxs)-1):
        idx1 = left_eyebrow_idxs[i]
        idx2 = left_eyebrow_idxs[i+1]
        x1 = func(face.landmark[idx1].x, image_width)
        y1 = func(face.landmark[idx1].y, image_height)
        x2 = func(face.landmark[idx2].x, image_width)
        y2 = func(face.landmark[idx2].y, image_height)

        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)

    for i in range(len(right_eyebrow_idxs)-1):
        idx1 = right_eyebrow_idxs[i]
        idx2 = right_eyebrow_idxs[i+1]
        x1 = func(face.landmark[idx1].x, image_width)
        y1 = func(face.landmark[idx1].y, image_height)
        x2 = func(face.landmark[idx2].x, image_width)
        y2 = func(face.landmark[idx2].y, image_height)

        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)

    for i in range(len(lip_idxs)-1):
        idx1 = lip_idxs[i]
        idx2 = lip_idxs[i+1]
        x1 = func(face.landmark[idx1].x, image_width)
        y1 = func(face.landmark[idx1].y, image_height)
        x2 = func(face.landmark[idx2].x, image_width)
        y2 = func(face.landmark[idx2].y, image_height)

        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
    
    return image
    

def drawC(image, face, image_width, image_height):
    nosex = func(face.landmark[1].x, image_width)
    nosey = func(face.landmark[1].y, image_height)
    face_size = func(np.sqrt((face.landmark[234].x - face.landmark[454].x)**2 + (face.landmark[234].y - face.landmark[454].y)**2) / 2 * 1.2, image_width)

    eye_width1 = func(np.sqrt((face.landmark[133].x - face.landmark[33].x)**2 + (face.landmark[133].y - face.landmark[33].y)**2) * 2, image_width)
    eye_height1 = func(np.sqrt((face.landmark[159].x - face.landmark[145].x)**2 + (face.landmark[159].y - face.landmark[145].y)**2) * 3, image_height)
    eye_width2 = func(np.sqrt((face.landmark[362].x - face.landmark[263].x)**2 + (face.landmark[362].y - face.landmark[263].y)**2) * 2, image_width)
    eye_height2 = func(np.sqrt((face.landmark[386].x - face.landmark[374].x)**2 + (face.landmark[386].y - face.landmark[374].y)**2) * 3, image_height)
    eye_angle1 = np.arctan2(face.landmark[33].y - face.landmark[133].y, face.landmark[33].x - face.landmark[133].x) * 180 / np.pi
    eye_angle2 = np.arctan2(face.landmark[263].y - face.landmark[362].y, face.landmark[263].x - face.landmark[362].x) * 180 / np.pi

    eye_center1x = func((face.landmark[133].x + face.landmark[33].x + face.landmark[159].x + face.landmark[145].x) / 4, image_width)
    eye_center1y = func((face.landmark[133].y + face.landmark[33].y + face.landmark[159].y + face.landmark[145].y) / 4, image_height)
    eye_center2x = func((face.landmark[362].x + face.landmark[263].x + face.landmark[386].x + face.landmark[374].x) / 4, image_width)
    eye_center2y = func((face.landmark[362].y + face.landmark[263].y + face.landmark[386].y + face.landmark[374].y) / 4, image_height)
    
    pupil1x = func(face.landmark[468].x, image_width)
    pupil1y = func(face.landmark[468].y, image_height)
    pupil2x = func(face.landmark[473].x, image_width)
    pupil2y = func(face.landmark[473].y, image_height)

    iris_size1a = func(np.sqrt((face.landmark[159].x - face.landmark[145].x)**2 + (face.landmark[159].y - face.landmark[145].y)**2), image_width)
    iris_size1b = int(iris_size1a / 2)
    iris_size2a = func(np.sqrt((face.landmark[386].x - face.landmark[374].x)**2 + (face.landmark[386].y - face.landmark[374].y)**2), image_width)
    iris_size2b = int(iris_size2a / 2)

    lip_width = func(np.sqrt((face.landmark[57].x - face.landmark[287].x)**2 + (face.landmark[57].y - face.landmark[287].y)**2), image_width)
    lip_height = func(np.sqrt((face.landmark[0].x - face.landmark[17].x)**2 + (face.landmark[0].y - face.landmark[17].y)**2), image_height)
    lip_angle = np.arctan2(face.landmark[57].y - face.landmark[287].y, face.landmark[57].x - face.landmark[287].x) * 180 / np.pi
    lip_centerx = func((face.landmark[57].x + face.landmark[287].x + face.landmark[0].x + face.landmark[17].x) / 4, image_width)
    lip_centery = func((face.landmark[57].y + face.landmark[287].y + face.landmark[0].y + face.landmark[17].y) / 4, image_height)

    cv2.circle(image, center=(nosex, nosey), radius=face_size, color=(135, 184, 222), thickness=-1)
    cv2.ellipse(image, ((eye_center1x, eye_center1y), (eye_width1, eye_height1), eye_angle1), (255, 255, 255), -1)
    cv2.ellipse(image, ((eye_center2x, eye_center2y), (eye_width2, eye_height2), eye_angle2), (255, 255, 255), -1)
    cv2.circle(image, center=(pupil1x, pupil1y), radius=iris_size1a, color=(150, 150, 0), thickness=-1)
    cv2.circle(image, center=(pupil1x, pupil1y), radius=iris_size1b, color=(0, 0, 0), thickness=-1)
    cv2.circle(image, center=(pupil2x, pupil2y), radius=iris_size2a, color=(150, 150, 0), thickness=-1)
    cv2.circle(image, center=(pupil2x, pupil2y), radius=iris_size2b, color=(0, 0, 0), thickness=-1)
    cv2.ellipse(image, ((lip_centerx, lip_centery), (lip_width, lip_height), lip_angle), (150, 150, 255), -1)
    
    return image


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

        if draw_pattern == "A":
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                   for landmark in face.landmark:               
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)

        elif draw_pattern == "B":
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    out_image = drawB(out_image, face, image_width, image_height) 

        elif draw_pattern == "C":
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    out_image = drawC(out_image, face, image_width, image_height) 

    return cv2.flip(out_image, 1)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def __init__(self) -> None:
        self.is_show_image = True
        self.draw_pattern = "A"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img, self.is_show_image, self.draw_pattern)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
