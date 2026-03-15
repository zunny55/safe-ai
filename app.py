import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.title("🦺 AI PPE Detection System")

# load models
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    frame = np.array(image)

    # detect person
    person_results = person_model(frame)

    # detect PPE
    ppe_results = ppe_model(frame, conf=0.1)

    helmet = False
    vest = False

    # check PPE detection
    for r in ppe_results:
        for box in r.boxes:

            label = ppe_model.names[int(box.cls[0])].lower()

            if "helmet" in label or "hat" in label:
                helmet = True

            if "vest" in label or "jacket" in label:
                vest = True

    # draw person box
    for r in person_results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if person_model.names[cls] == "person":

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                if helmet and vest:
                    cv2.putText(frame,"SAFE",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,(0,255,0),2)
                else:
                    cv2.putText(frame,"NO PPE",(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,(255,0,0),2)

    st.image(frame, channels="BGR")

    if not helmet:
        st.error("⚠ Worker without helmet detected")

    if not vest:
        st.error("⚠ Worker without vest detected")

    if helmet and vest:
        st.success("✅ All workers wearing PPE")
