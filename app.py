import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI PPE Detection", layout="centered")

st.title("🦺 AI PPE Detection System")
st.write("Upload an image to detect helmet and safety vest")

# -----------------------
# Load models
# -----------------------
@st.cache_resource
def load_models():
    person_model = YOLO("yolov8n.pt")
    ppe_model = YOLO("best.pt")
    return person_model, ppe_model

person_model, ppe_model = load_models()

# -----------------------
# Upload image
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Convert to RGB
    image = image.convert("RGB")

    # Convert to numpy
    frame = np.array(image)

    # Safety fix (remove alpha channel if exists)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    helmet_detected = False
    vest_detected = False

    # -----------------------
    # Detect PPE
    # -----------------------
    results = ppe_model(frame, conf=0.25)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls)
            label = ppe_model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if "helmet" in label:
                helmet_detected = True
                color = (0,255,0)

            elif "vest" in label or "jacket" in label:
                vest_detected = True
                color = (255,255,0)

            else:
                color = (255,0,0)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

            cv2.putText(
                frame,
                label,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # -----------------------
    # Show image
    # -----------------------
    st.image(frame, caption="Detection Result", use_column_width=True)

    # -----------------------
    # Safety Status
    # -----------------------
    if helmet_detected and vest_detected:
        st.success("✅ Worker wearing helmet and vest")

    else:

        if not helmet_detected:
            st.error("⚠ Helmet not detected")

        if not vest_detected:
            st.error("⚠ Safety vest not detected")
