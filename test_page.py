import streamlit as st
import pandas as pd
import base64
import cv2 as cv
import numpy as np
import av
import mediapipe as mp
import os
import tempfile
from cvzone.HandTrackingModule import HandDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from glob import glob




detector = HandDetector(detectionCon=0.8, maxHands=2)

offset = 50
size = (300, 300)
hand_features = []


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('bg_5hands.jpg')

st.header(':gray[_Welcome to NyoKi Classifier_]')
st.subheader('Hand Sign Recognition Application (Word Level)')
activities = ["Home", "Webcam Hand Detection", "Video File Hand Detection", "Thanks"]
choice_s = st.sidebar.selectbox("Select Activity <3", activities)


@st.cache
def process(image):
    hands_crop, image_crop = detector.findHands(image)
    if hands_crop:
        hand_crop = hands_crop[0]
        x, y, w, h = hand_crop['bbox']
        img_crop = image[y - offset: y + h + offset, x - offset: x + w + offset]

    return image


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


if choice_s == "Home":
    html_temp_home1 = """<div style="background-color:#454545;padding:10px">
                              <h4 style="color:white;text-align:center;">
                              Hand Sign recognition application using OpenCV, Streamlit.
                              </h4>
                              </div>
                              </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)

elif choice_s == "Webcam Hand Detection":
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
elif choice_s == "Video File Hand Detection":
    uploaded_files = st.file_uploader("Choose a video file.", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # bytes_data = upload_file.getvalue()
            # st.write(bytes_data)

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vf = cv.VideoCapture(tfile.name)
            result = st.empty()
            while vf.isOpened():
                ret, frame = vf.read()

                if not ret:
                    #print("Can't receive frame (stream end?). Exiting ...")
                    st.write("Can't receive frame (stream end?). Exiting ...")
                    break

                video = process(frame)
                image = cv.cvtColor(video, cv.COLOR_BGR2RGB)
                result.image(image)



