import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------
# Title
# -----------------------
st.title("SafeVision AI - PPE Detection")
st.write("Detect Helmet and Safety Vest")

# -----------------------
# Load model
# -----------------------
model = YOLO("best.pt")

# -----------------------
# Upload Image
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    results = model(frame)

    helmet_missing = False
    vest_missing = False

    for r in results:

        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            color = (0,255,0)

            if label == "helmet":
                color = (0,255,0)

            elif label == "vest":
                color = (255,200,0)

            elif label == "person":
                color = (0,200,255)

            elif label == "no_helmet":
                color = (0,0,255)
                helmet_missing = True

            elif label == "no_vest":
                color = (0,0,255)
                vest_missing = True

            else:
                color = (200,200,200)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            text = f"{label} {conf:.2f}"

            cv2.putText(
                frame,
                text,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # -----------------------
    # Alerts
    # -----------------------

    if helmet_missing:
        st.error("⚠ Worker without helmet detected")

    if vest_missing:
        st.error("⚠ Worker without safety vest detected")

    if not helmet_missing and not vest_missing:
        st.success("✅ All workers wearing PPE")

    st.image(frame, channels="BGR")
