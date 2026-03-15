import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI PPE Detection", layout="centered")

st.title("🦺 AI PPE Detection System")
st.write("Upload image to detect helmet and safety vest")

# โหลดโมเดล
@st.cache_resource
def load_models():
    person_model = YOLO("yolov8n.pt")
    ppe_model = YOLO("best.pt")
    return person_model, ppe_model

person_model, ppe_model = load_models()

# upload image
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    helmet_detected = False
    vest_detected = False

    # detect person
    person_results = person_model(frame)

    # detect PPE
    ppe_results = ppe_model(frame, conf=0.1)

    # ตรวจ PPE
    for r in ppe_results:
        for box in r.boxes:
            cls = int(box.cls)
            label = ppe_model.names[cls].lower()

            if "helmet" in label or "hat" in label:
                helmet_detected = True

            if "vest" in label or "jacket" in label:
                vest_detected = True

    # วาด box คน
    for r in person_results:
        for box in r.boxes:

            cls = int(box.cls)

            if person_model.names[cls] == "person":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if helmet_detected and vest_detected:
                    color = (0, 255, 0)
                    text = "SAFE"
                else:
                    color = (255, 0, 0)
                    text = "NO PPE"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2
                )

    st.image(frame, caption="Detection Result")

    # แจ้งเตือน
    if not helmet_detected:
        st.error("⚠ Worker without helmet detected")

    if not vest_detected:
        st.error("⚠ Worker without vest detected")

    if helmet_detected and vest_detected:
        st.success("✅ All workers wearing PPE")
