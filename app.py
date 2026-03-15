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

    # ===== ใช้แค่ครึ่งบนของภาพ =====
    upper_half = img[0:int(h*0.5), :]

    hsv = cv2.cvtColor(upper_half, cv2.COLOR_RGB2HSV)

    # ===== สีหมวก =====
    helmet_yellow = cv2.inRange(hsv, (20,120,120), (35,255,255))
    helmet_orange = cv2.inRange(hsv, (5,120,120), (15,255,255))
    helmet_white = cv2.inRange(hsv, (0,0,220), (180,40,255))

    helmet_pixels = (
        np.sum(helmet_yellow > 0) +
        np.sum(helmet_orange > 0) +
        np.sum(helmet_white > 0)
    )

    # ===== สีเสื้อกั๊ก =====
    vest_yellow = cv2.inRange(hsv, (20,120,120), (35,255,255))
    vest_orange = cv2.inRange(hsv, (5,120,120), (15,255,255))

    vest_pixels = (
        np.sum(vest_yellow > 0) +
        np.sum(vest_orange > 0)
    )

    st.subheader("Result")

    if helmet_pixels > 6000:
        st.success("🪖 Helmet detected")
    else:
        st.error("❌ No helmet")

    if vest_pixels > 6000:
        st.success("🦺 Safety vest detected")
    else:
        st.error("❌ No safety vest")
