import os
import cv2
import math
import time
import numpy as np
from collections import defaultdict, deque
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI()

# হালকা ও দ্রুতগতির YOLOv8 Nano মডেল (রেন্ডারের ফ্রি র‍্যামের জন্য উপযোগী)
model = YOLO("yolov8n.pt")

# ট্র্যাকিং ও স্পিড ডেটা
track_history = defaultdict(lambda: deque(maxlen=10))
speed_history = defaultdict(float)

# পিক্সেল টু মিটার স্কেল (ক্যালিব্রেশন)
PIXELS_PER_METER = 8.5 

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="bn">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MAX — Camara hack</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', monospace; }
    body { background: #030712; color: #fff; display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 20px; }
    .badge { background: linear-gradient(135deg, #00f0ff, #0066ff); color: #000; font-weight: 900; font-size: 12px; padding: 3px 12px; border-radius: 20px; letter-spacing: 2px; }
    h1 { font-size: 1.8rem; margin: 8px 0; text-transform: uppercase; color: #fff; text-shadow: 0 0 15px rgba(0,240,255,0.4); }
    .dev-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(255,0,85,0.15); border: 1px solid #ff0055; color: #ff3377; text-decoration: none; font-weight: bold; font-size: 13px; padding: 6px 16px; border-radius: 30px; box-shadow: 0 0 12px rgba(255,0,85,0.4); margin-bottom: 20px; transition: 0.2s; }
    .dev-badge:hover { background: #ff0055; color: #fff; }
    .card { background: #0f172a; border: 1px solid #1e293b; padding: 20px; border-radius: 12px; width: 100%; max-width: 650px; text-align: center; }
    .upload-btn { display: inline-block; background: #00ff66; color: #000; padding: 12px 24px; border-radius: 8px; font-weight: bold; cursor: pointer; margin-top: 10px; font-size: 14px; }
    input[type="file"] { display: none; }
    #videoContainer { margin-top: 20px; width: 100%; max-width: 750px; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: none; }
    #streamView { width: 100%; display: block; }
    footer { margin-top: auto; padding: 20px; font-size: 12px; color: #64748b; }
  </style>
</head>
<body>
  <div class="badge">MAX</div>
  <h1>Camara hack</h1>
  <a href="https://t.me/dngrmahin" target="_blank" class="dev-badge">DEVELOPER: MAHIN SIR</a>

  <div class="card">
    <p style="color:#94a3b8; font-size:14px; margin-bottom:12px;">পাইথন YOLOv8 এবং অপটিক্যাল স্পিড ট্র্যাকিং সার্ভার</p>
    <label class="upload-btn" for="fileInput">📁 ভিডিও ফাইল নির্বাচন করুন</label>
    <input type="file" id="fileInput" accept="video/*" onchange="uploadVideo()">
  </div>

  <div id="videoContainer">
    <img id="streamView" src="" alt="Processing Video Stream...">
  </div>

  <footer>
    <div id="copyYear">© <span id="yr">...</span> MAX — DEVELOPER MAHIN SIR</div>
  </footer>

  <script>
    // লাইভ অনলাইন টাইম API থেকে সাল সিঙ্ক
    fetch('https://worldtimeapi.org/api/timezone/Etc/UTC')
      .then(res => res.json())
      .then(d => { document.getElementById('yr').innerText = new Date(d.utc_datetime).getFullYear(); })
      .catch(() => { document.getElementById('yr').innerText = new Date().getFullYear(); });

    function uploadVideo() {
      const file = document.getElementById('fileInput').files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      document.getElementById('videoContainer').style.display = "block";
      document.getElementById('streamView').src = "/process_stream?filename=" + encodeURIComponent(file.name);

      fetch('/upload', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
          document.getElementById('streamView').src = "/stream/" + data.filename;
        });
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ফ্রেম সাইজ ৬৪০ এ রিসাইজ করা (রেন্ডারের স্পিড বজায় রাখতে)
        frame = cv2.resize(frame, (640, 360))
        h, w, _ = frame.shape
        curr_time = time.time()

        # YOLO ও ByteTrack দিয়ে অবজেক্ট ট্র্যাকিং
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = y2  # গাড়ির নিচের পয়েন্ট

                # গতি হিসাব
                history = track_history[track_id]
                history.append((cx, cy, curr_time))

                speed_kph = speed_history[track_id]
                if len(history) >= 2:
                    p1 = history[0]
                    p2 = history[-1]
                    dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    dt = p2[2] - p1[2]
                    if dt > 0.05:
                        meters = dist_px / PIXELS_PER_METER
                        inst_kph = (meters / dt) * 3.6
                        speed_kph = speed_kph * 0.7 + inst_kph * 0.3 if speed_kph > 0 else inst_kph
                        speed_history[track_id] = speed_kph

                # কালার কোডিং (ভিডিওর মতো)
                color = (0, 255, 0) # Green (<60)
                if 60 <= speed_kph <= 100:
                    color = (0, 255, 255) # Yellow
                elif speed_kph > 100:
                    color = (0, 0, 255) # Red

                label_name = model.names[cls].upper()
                label = f"{label_name} [{int(speed_kph)} km/h]"

                # বাউন্ডিং বক্স আঁকা
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # লেবেল ব্যাকগ্রাউন্ড ও টেক্সট
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 6, y1), (15, 23, 42), -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # স্পিড বার
                bar_w = int(min(speed_kph / 120, 1.0) * (x2 - x1))
                cv2.rectangle(frame, (x1, y2 - 4), (x1 + bar_w, y2), color, -1)

        # ভিডিওর মতো নিচে বামে লেজেন্ড (Legend)
        cv2.rectangle(frame, (10, h - 65), (140, h - 10), (10, 15, 25), -1)
        cv2.circle(frame, (20, h - 50), 4, (0, 255, 0), -1)
        cv2.putText(frame, "< 60 km/h", (32, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.circle(frame, (20, h - 35), 4, (0, 255, 255), -1)
        cv2.putText(frame, "60-100 km/h", (32, h - 31), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.circle(frame, (20, h - 20), 4, (0, 0, 255), -1)
        cv2.putText(frame, "> 100 km/h", (32, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # ফ্রেম এনকোড
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/stream/{filename}")
def stream_video(filename: str):
    file_path = os.path.join("uploads", filename)
    return StreamingResponse(generate_frames(file_path), media_type="multipart/x-mixed-replace; boundary=frame")
