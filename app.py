import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🦺 PPE Color Detector")

file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:

    image = Image.open(file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    h, w, _ = img.shape

    # ===== แบ่งพื้นที่ตรวจ =====

    helmet_area = img[0:int(h*0.25), :]          # ส่วนหัว
    vest_area = img[int(h*0.35):int(h*0.65), :]  # ส่วนลำตัว

    hsv_helmet = cv2.cvtColor(helmet_area, cv2.COLOR_RGB2HSV)
    hsv_vest = cv2.cvtColor(vest_area, cv2.COLOR_RGB2HSV)

    # ===== สีหมวก =====

    helmet_yellow = cv2.inRange(hsv_helmet, (20,120,120), (35,255,255))
    helmet_orange = cv2.inRange(hsv_helmet, (5,120,120), (15,255,255))
    helmet_white = cv2.inRange(hsv_helmet, (0,0,220), (180,40,255))

    helmet_pixels = (
        np.sum(helmet_yellow > 0) +
        np.sum(helmet_orange > 0) +
        np.sum(helmet_white > 0)
    )

    # ===== สีเสื้อกั๊ก =====

    vest_yellow = cv2.inRange(hsv_vest, (20,120,120), (35,255,255))
    vest_orange = cv2.inRange(hsv_vest, (5,120,120), (15,255,255))

    vest_pixels = (
        np.sum(vest_yellow > 0) +
        np.sum(vest_orange > 0)
    )

    st.subheader("Result")

    if helmet_pixels > 5000:
        st.success("🪖 Helmet detected")
    else:
        st.error("❌ No helmet")

    if vest_pixels > 5000:
        st.success("🦺 Safety vest detected")
    else:
        st.error("❌ No safety vest")
