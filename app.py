import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

st.title("🦺 PPE Detection System")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    results = model(frame, conf=0.25)

    helmet = False
    vest = False

    for r in results:
        for box in r.boxes:

            cls = int(box.cls)
            label = model.names[cls]

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            if label.lower() == "helmet":
                helmet = True
                color = (0,255,0)

            elif label.lower() == "vest":
                vest = True
                color = (255,255,0)

            elif label.lower() == "person":
                color = (255,0,0)

            else:
                color = (0,0,255)

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

    st.image(frame)

    if helmet and vest:
        st.success("SAFE")

    else:
        st.error("NO PPE")
