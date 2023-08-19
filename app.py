import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
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
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        cv2.circle(out_image, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
                        cv2.circle(out_image, center=(x, y), radius=1, color=(255, 255, 255), thickness=-1)

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
    out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, frame_fps, (width, height), isColor = False)
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_image = process(frame, is_show_image, draw_pattern)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ##<< Generates a grayscale (thus only one 2d-array)
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
    col2.header("Output from OpenCV (MPEG-4)")
    col2.video(temp_file_result)
    col2.header("After conversion to H264")
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
    webrtc_ctx.video_processor.draw_pattern = st.radio("select draw pattern", ["A", "None"], key="A", horizontal=True)
