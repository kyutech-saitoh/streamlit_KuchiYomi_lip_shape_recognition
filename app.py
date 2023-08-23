import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import math
import numpy as np
import av
import mediapipe as mp
import subprocess

import tempfile

st.title("Streamlit App: Face motion by MediaPipe")
st.write("Kyutech, Saitoh-lab")

video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'

# left eye contour
landmark_left_eye_points = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155]
# right eye contour
landmark_right_eye_points = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

size_faceROI =  256

def func(value1, value2):
    return int(value1 * value2)

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
        z0 = p0[2]
        x1 = mat_rot[0][0] * x0 + mat_rot[1][0] * y0 + mat_rot[0][2]
        y1 = mat_rot[0][1] * x0 + mat_rot[1][1] * y0 + mat_rot[1][2]
        x2 = x1 + mat_tra[0][2]
        y2 = y1 + mat_tra[1][2]

        face_points1.append((x2, y2, z0))

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
                   for landmark in face.landmark:
                        x = func(landmark.x, image_width)
                        y = func(landmark.y, image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)
                        """
                        if results.multi_face_landmarks:
                        for face in results.multi_face_landmarks:
                        points = []
                        for landmark in face.landmark:
                        x = func(landmark.x, image_width)
                        y = func(landmark.y, image_height)
                        
                        #                        points.append((x, y))
                        """
                        """
                        rect_faceROI, normalized_image_faceROI, new_points_faceROI = faceROI_extraction(image, points)
                        faceROI = normalized_image_faceROI[rect_faceROI[1]: rect_faceROI[3], rect_faceROI[0]: rect_faceROI[2]]
                        
                        out_image[0: size_faceROI, 0: size_faceROI] = faceROI
                        """            

    return cv2.flip(out_image, 1)

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

if video_data:
    # save uploaded video to disc
    write_bytesio_to_file(temp_file_to_save, video_data)

    # read it with cv2.VideoCapture(), 
    # so now we can process it with OpenCV functions
    cap = cv2.VideoCapture(temp_file_to_save)

    # grab some parameters of video to use them for writing a new, processed video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int
    st.write(width, height, frame_fps)
    
    # specify a writer to write a processed video to a disk frame by frame
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height))
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_image = process(frame, True, "A")

#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##<< Generates a grayscale (thus only one 2d-array)
        out_mp4.write(out_image)
    
    ## Close video files
    out_mp4.release()
    cap.release()

    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    convertedVideo = "./testh264.mp4"
    subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))
    
    ## Show results
    col1, col2 = st.columns(2)
    col1.header("Original Video")
    col1.video(temp_file_to_save)
#    col2.header("Output from OpenCV (MPEG-4)")
#    col2.video(temp_file_result)
#    col2.header("After conversion to H264")
    col2.header("Output")
    col2.video(convertedVideo)


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
    webrtc_ctx.video_processor.draw_pattern = st.radio("select draw pattern", ["A", "B", "None"], key="A", horizontal=True)
