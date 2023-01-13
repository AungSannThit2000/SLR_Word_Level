import cv2
import numpy as np
import av
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

detector = HandDetector(detectionCon=0.8, maxHands=2)

offset = 50
size = (300, 300)
hand_features = []
# def process(image):
#     hands_crop, image_crop = detector.findHands(image.copy())
#     if hands_crop:
#         hand_crop = hands_crop[0]
#         x, y, w, h = hand_crop['bbox']
#         img_crop = image[y - offset: y + h + offset, x - offset: x + w + offset]
#
#         img_crop = cv2.resize(img_crop, size)  # Tune the image size
#         hands, img = detector.findHands(img_crop.copy())
#         if hands:
#             hand = hands[0]
#             hand_lm = hand['lmList']
#             hand_lm = np.array(hand_lm).flatten()
#             x, y, w, h = hand['bbox']
#             hand_features.append(hand_lm)
#
#         cv2.imshow("Cropped Image", img)
#     cv2.imshow("Original Image", image_crop)
#     return image_crop

def process(image):
    hands_crop, image_crop = detector.findHands(image)
    if hands_crop:
        hand_crop = hands_crop[0]
        x, y, w, h = hand_crop['bbox']
        img_crop = image[y - offset: y + h + offset, x - offset: x + w + offset]


    #cv2.imshow("Original Image", image_crop)
    return image_crop

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
