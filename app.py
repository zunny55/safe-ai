import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="SafeVision AI", layout="wide")

st.title("🦺 SafeVision AI - PPE Detection")

# โหลดโมเดล
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # โหลดรูป
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    # ตรวจจับคน
    results = person_model(frame)

    helmet_missing = False
    vest_missing = False

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if person_model.names[cls] != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ขยายกรอบ
            pad = 30
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            person_crop = frame[y1:y2, x1:x2]

            # ตรวจ PPE
            ppe_results = ppe_model(person_crop)

            helmet = False
            vest = False

            for pr in ppe_results:
                for pbox in pr.boxes:

                    label = ppe_model.names[int(pbox.cls[0])].lower()

                    if label in ["helmet","hardhat","hat"]:
                        helmet = True

                    if label in ["vest","safety vest","safety_vest","jacket"]:
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

            # วาดกรอบ
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            # ใส่ข้อความ
            cv2.putText(
                frame,
                text,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

    st.image(frame)

    # แสดงผลแจ้งเตือน
    if helmet_missing:
        st.error("⚠ Worker without helmet detected")

    if vest_missing:
        st.error("⚠ Worker without vest detected")

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")
