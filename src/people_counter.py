import json
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import can
import time
import threading

model = YOLO("yolov8n.pt")

tracker_cam1 = DeepSort(max_age=30)
tracker_cam2 = DeepSort(max_age=30)
line_position = 300

counters = {
    "cam1": {"entry": 0, "exit": 0, "track_memory": {}},
    "cam2": {"entry": 0, "exit": 0, "track_memory": {}}
}

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

print("[INFO] Starting Multi-Camera People Counter...")

# Initialize CAN interface
can_bus = can.Bus(interface='virtual', channel='vcan0', receive_own_messages=True)


def process_frame(frame, tracker, counter_dict, cam_name):
    results = model(frame, verbose=False)[0]
    detections = []

    for r in results.boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        if model.names[cls] == 'person' and conf >= 0.70:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        center_y = int((t + b) / 2)

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f'{cam_name} ID:{track_id}', (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if track_id not in counter_dict["track_memory"]:
            counter_dict["track_memory"][track_id] = center_y
        else:
            prev_y = counter_dict["track_memory"][track_id]
            if prev_y < line_position <= center_y:
                counter_dict["entry"] += 1
                counter_dict["track_memory"][track_id] = center_y
            elif prev_y > line_position >= center_y:
                counter_dict["exit"] += 1
                counter_dict["track_memory"][track_id] = center_y

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)

    cv2.putText(frame, f"Entries: {counter_dict['entry']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits:   {counter_dict['exit']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def update_json_and_can():
    while True:
        data = {
            "CAM1": {
                "Entries": counters["cam1"]["entry"],
                "Exits": counters["cam1"]["exit"]
            },
            "CAM2": {
                "Entries": counters["cam2"]["entry"],
                "Exits": counters["cam2"]["exit"]
            },
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save JSON file
        with open("people_counts.json", "w") as f:
            json.dump(data, f, indent=4)

        print("[JSON + CAN] Data saved and sending on CAN...")

        # Pack CAN message
        try:
            msg_data = f"{data['CAM1']['Entries']},{data['CAM1']['Exits']},{data['CAM2']['Entries']},{data['CAM2']['Exits']}"
            byte_data = msg_data.encode('utf-8')[:8]  # CAN allows max 8 bytes
            msg = can.Message(arbitration_id=0x123, data=byte_data, is_extended_id=False)
            can_bus.send(msg)
        except can.CanError:
            print("[CAN ERROR] Message not sent")

        time.sleep(5)  # Update every 5 seconds

# Start background thread for JSON+CAN update
t = threading.Thread(target=update_json_and_can, daemon=True)
t.start()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        print("[ERROR] Failed to grab frames.")
        break

    if ret1:
        frame1 = process_frame(frame1, tracker_cam1, counters["cam1"], "CAM1")
        cv2.imshow("Camera 1", frame1)

    if ret2:
        frame2 = process_frame(frame2, tracker_cam2, counters["cam2"], "CAM2")
        cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
