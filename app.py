import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("SafeVision AI - Industrial Safety Monitoring")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an image from factory",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    frame = np.array(image)

    results = model(frame)

    danger_x1 = 200
    danger_y1 = 150
    danger_x2 = 450
    danger_y2 = 350

    cv2.rectangle(frame,(danger_x1,danger_y1),(danger_x2,danger_y2),(0,0,255),2)
    cv2.putText(frame,"DANGER ZONE",(danger_x1,danger_y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    for r in results:

        for box in r.boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                if danger_x1 < cx < danger_x2 and danger_y1 < cy < danger_y2:

                    cv2.putText(frame,
                                "ALERT: Worker in Danger Zone!",
                                (30,50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,0,255),
                                3)

    st.image(frame, caption="AI Detection Result", use_column_width=True)
