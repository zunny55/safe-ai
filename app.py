import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="SafeVision AI", layout="centered")

st.title("🦺 SafeVision AI")
st.subheader("PPE Detection System")

# โหลดโมเดลที่ train แล้ว
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    results = model(frame, conf=0.4)

    helmet_missing = False
    vest_missing = False

    for r in results:
        boxes = r.boxes

        for box in boxes:

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            color = (0,255,0)

            if label == "helmet":
                color = (0,255,0)

            elif label == "vest":
                color = (255,200,0)

            elif label == "no_helmet":
                color = (0,0,255)
                helmet_missing = True

            elif label == "no_vest":
                color = (0,0,255)
                vest_missing = True

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    st.image(frame, channels="BGR")

    if helmet_missing:
        st.error("⚠ Worker without helmet detected")

    if vest_missing:
        st.error("⚠ Worker without vest detected")

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")
