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

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    person_results = person_model(frame)

    helmet_missing = False
    vest_missing = False

    for r in person_results:

        for box in r.boxes:

            cls = int(box.cls[0])

            if person_model.names[cls] != "person":
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            # ขยายกรอบเพื่อไม่ให้ตัดหมวก
            pad = 30

            y1 = max(0,y1-pad)
            x1 = max(0,x1-pad)
            y2 = min(frame.shape[0],y2+pad)
            x2 = min(frame.shape[1],x2+pad)

            person_crop = frame[y1:y2, x1:x2]

            ppe_results = ppe_model(person_crop)

            helmet = False
            vest = False

            for pr in ppe_results:

                for pbox in pr.boxes:

                    label = ppe_model.names[int(pbox.cls[0])]

                    # ตรวจหลายชื่อเผื่อ dataset ใช้ชื่อไม่เหมือนกัน
                    if label.lower() in ["helmet","hardhat","hat"]:
                        helmet = True

                    if label.lower() in ["vest","safety vest","safety_vest","jacket"]:
                        vest = True

            label_text = "SAFE"
            color = (0,255,0)

            if not helmet:
                helmet_missing = True
                label_text = "NO HELMET"
                color = (0,0,255)

            if not vest:
                vest_missing = True
                label_text = "NO VEST"
                color = (0,0,255)

            if not helmet and not vest:
                label_text = "NO PPE"

            # วาดกรอบ
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            # ใส่ข้อความบนกรอบ
            cv2.putText(frame,
                        label_text,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2)

    st.image(frame, channels="BGR")

    if helmet_missing:
        st.error("⚠ Worker without helmet detected")

    if vest_missing:
        st.error("⚠ Worker without vest detected")

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")success("✅ All workers wearing PPE")
