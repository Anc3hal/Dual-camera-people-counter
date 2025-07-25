# Dual-camera-people-counter
# Multi-Camera People Counter using YOLOv8 and DeepSORT

This project is a real-time **people counting system** using **YOLOv8 object detection** and **DeepSORT object tracking** on two camera feeds (USB or IP cams). It tracks individuals across a virtual line to count **entries and exits** separately for each camera.

---

## Features

- Real-time person detection using YOLOv8 (`yolov8n.pt`)
- Multi-camera support (2 video sources)
- Entry and Exit line counting logic
- Unique ID tracking using DeepSORT
- Live video display with count overlays
- Logs events like Entry and Exit with ID

---

## Requirements

Install the following Python packages before running the project:

```bash
pip install ultralytics
pip install opencv-python
pip install deep_sort_realtime
