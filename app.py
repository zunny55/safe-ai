import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# หน้าเว็บ
st.title("SafeVision AI - Industrial Safety Monitoring")
st.write("Upload a factory image to detect workers and safety violations")

# โหลดโมเดล
model = YOLO("yolov8n.pt")

# อัปโหลดรูป
uploaded_file = st.file_uploader(
    "Upload factory image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    frame = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Start AI Detection"):

        results = model(frame)

        # ขนาดภาพ
        h, w, _ = frame.shape

        # Danger Zone (กลางภาพ)
        danger_x1 = int(w * 0.3)
        danger_y1 = int(h * 0.3)
        danger_x2 = int(w * 0.7)
        danger_y2 = int(h * 0.7)

        # วาด Danger Zone
        cv2.rectangle(frame,(danger_x1,danger_y1),(danger_x2,danger_y2),(0,0,255),2)
        cv2.putText(frame,
                    "DANGER ZONE",
                    (danger_x1,danger_y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2)

        worker_count = 0
        alert = False

        for r in results:
            for box in r.boxes:

                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # ลด detection ผิด
                if conf < 0.6:
                    continue

                label = model.names[cls]

                # ตรวจเฉพาะคน
                if label != "person":
                    continue

                worker_count += 1

                # วาดกรอบ
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                text = f"{label} {conf:.2f}"

                cv2.putText(frame,
                            text,
                            (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

                # จุดกลางคน
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                # ตรวจ Danger Zone
                if danger_x1 < cx < danger_x2 and danger_y1 < cy < danger_y2:
                    alert = True

        # แจ้งเตือน
        if alert:
            cv2.putText(frame,
                        "ALERT: Worker in Danger Zone!",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

        # แสดงผลลัพธ์
        st.image(frame, caption="AI Detection Result", use_column_width=True)

        st.write(f"Workers detected: {worker_count}")

        if alert:
            st.error("Safety Alert: Worker detected inside danger zone")
        else:
            st.success("No worker in danger zone")
