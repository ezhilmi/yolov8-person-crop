
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import imutils
import torch
import cv2
import cvzone
import os
from datetime import datetime

vehicle_detector= YOLO("yolov8n.pt")

# Initialize video capture
# video = cv2.VideoCapture(0)
CCTV_RTSP = 'rtsp://admin:senatraffic1234*@192.168.1.65/channels/101'
VIDEOPATH = 'CP038_JALAN_RAJA_CHULAN_07865E78_1686205144_1.mp4'
video = cv2.VideoCapture(VIDEOPATH)

fpsReader = cvzone.FPS()

print(torch.cuda.is_available())

#Getting classes
all_classes = vehicle_detector.model.names
# class_names = ['car', 'motorcycle', 'bus', 'truck']
print(all_classes)

while True:

    # # Read a new frame
    ret, frame = video.read()

    # Check if frame is read successfully
    if not ret:
    
        continue
    
    ###FPS
    fps, frame = fpsReader.update(frame, pos=(15,30), color=(0,255,0), scale = 2, thickness=3)

    frame = imutils.resize(frame, width=1680)
    annotator = Annotator(frame)

    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vehicle_detections = vehicle_detector.track(frame,save_crop= False,persist=True, device=0, tracker='bytetrack.yaml')
    
    for result in vehicle_detections:
        boxes = result.boxes.cpu().numpy()
        # plotted = result[0].plot()

        for box in boxes:
            (x, y, w, h) = box.xyxy[0]
            # cx = int((x + w)/ 2)
            # cy = int((y + h)/2)
            # cv2.circle(frame, (cx, cy), 3, (0 ,255, 0), -1)
            b = box.xyxy[0].astype(int)
            c = box.cls
            id = int(box.id[0]) if box.id is not None else 0

            annotator.box_label(b, f"ID:{id} {result.names[int(c)]} {float(box.conf):.2}", color=(255,0,0))

    # for i, box in enumerate(boxes):

    #     b = box.xyxy[0].astype(int)
    #     c = box.cls
    #     id = int(box.id[0]) if box.id is not None else 0
    #     print(id)

        # annotator.box_label(b, f"{result.names[int(c)]} {float(box.conf):.2}", color=(255,0,0))
        # annotator.box_label(b, f"ID:{id} {result.names[int(c)]} {float(box.conf):.2}", color=(255,0,0))
        
    cv2.imshow('Vehicle Detect', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
