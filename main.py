from ultralytics import YOLO
import cv2
import numpy as np

# โหลดโมเดล
model = YOLO("yolov8n.pt")

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# กำหนด Danger Zone (สี่เหลี่ยม)
danger_x1 = 200
danger_y1 = 150
danger_x2 = 450
danger_y2 = 350

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    # วาด Danger Zone
    cv2.rectangle(frame,(danger_x1,danger_y1),(danger_x2,danger_y2),(0,0,255),2)
    cv2.putText(frame,"DANGER ZONE",(danger_x1,danger_y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    helmet_detected = False

    for r in results:

        boxes = r.boxes

        for box in boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # ตรวจจับคน
            if label == "person":

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,"Worker",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

                # ตรวจ Danger Zone
                if danger_x1 < cx < danger_x2 and danger_y1 < cy < danger_y2:

                    cv2.putText(frame,
                                "ALERT: Worker in Danger Zone!",
                                (30,50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,0,255),
                                3)

            # ตรวจหมวก (จำลอง)
            if label == "helmet":

                helmet_detected = True

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                cv2.putText(frame,"Helmet",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255,0,0),
                            2)

    # แจ้งเตือนถ้าไม่มีหมวก
    if not helmet_detected:

        cv2.putText(frame,
                    "WARNING: No Helmet Detected",
                    (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    cv2.imshow("SafeVision AI",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
import streamlit as st
import cv2
from ultralytics import YOLO

st.title("SafeVision AI - Industrial Safety Monitoring")

model = YOLO("yolov8n.pt")

run = st.checkbox("Start Camera")

camera = cv2.VideoCapture(0)

frame_window = st.image([])

while run:
    ret, frame = camera.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    frame_window.image(annotated_frame)

camera.release()
