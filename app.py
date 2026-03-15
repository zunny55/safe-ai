import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

st.title("🦺 SafeVision AI")

# โมเดลตรวจคน
person_model = YOLO("yolov8n.pt")

# โมเดล PPE ที่คุณ train
ppe_model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    person_results = person_model(frame)

    helmet_missing = False
    vest_missing = False

    for r in person_results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if person_model.names[cls] != "person":
                continue

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            person_crop = frame[y1:y2, x1:x2]

            ppe_results = ppe_model(person_crop)

            helmet = False
            vest = False

            for pr in ppe_results:
                for pbox in pr.boxes:

                    label = ppe_model.names[int(pbox.cls[0])]

                    if label == "helmet":
                        helmet = True

                    if label == "vest":
                        vest = True

            if not helmet:
                helmet_missing = True

            if not vest:
                vest_missing = True

            color = (0,255,0)

            if not helmet or not vest:
                color = (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

    st.image(frame)

    if helmet_missing:
        st.error("⚠ Worker without helmet detected")

    if vest_missing:
        st.error("⚠ Worker without vest detected")

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")
