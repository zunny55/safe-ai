import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="SafeVision AI", layout="wide")

st.title("🦺 SafeVision AI - PPE Detection")

# โหลดโมเดล
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    # ตรวจคน
    person_results = person_model(frame)

    # ตรวจ PPE จากภาพเต็ม
    ppe_results = ppe_model(frame)

    helmet_boxes = []
    vest_boxes = []

    # เก็บตำแหน่ง PPE
    for r in ppe_results:
        for box in r.boxes:

            label = ppe_model.names[int(box.cls[0])].lower()
            x1,y1,x2,y2 = map(int,box.xyxy[0])

            if label in ["helmet","hardhat","hat"]:
                helmet_boxes.append((x1,y1,x2,y2))

            if label in ["vest","safety vest","safety_vest","jacket"]:
                vest_boxes.append((x1,y1,x2,y2))

    helmet_missing = False
    vest_missing = False

    for r in person_results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if person_model.names[cls] != "person":
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            helmet = False
            vest = False

            # เช็คว่าหมวกอยู่ในกรอบคนไหม
            for hx1,hy1,hx2,hy2 in helmet_boxes:
                if hx1 > x1 and hy1 > y1 and hx2 < x2 and hy2 < y2:
                    helmet = True

            # เช็คเสื้อ
            for vx1,vy1,vx2,vy2 in vest_boxes:
                if vx1 > x1 and vy1 > y1 and vx2 < x2 and vy2 < y2:
                    vest = True

            color = (0,255,0)
            text = "SAFE"

            if not helmet:
                helmet_missing = True
                color = (0,0,255)
                text = "NO HELMET"

            if not vest:
                vest_missing = True
                color = (0,0,255)
                text = "NO VEST"

            if not helmet and not vest:
                text = "NO PPE"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            cv2.putText(frame,
                        text,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2)

    st.image(frame)

    if helmet_missing:
        st.error("⚠ Worker without helmet detected")

    if vest_missing:
        st.error("⚠ Worker without vest detected")

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")
