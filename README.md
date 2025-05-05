# 🚦 Automatic Traffic Violation Detection System

This project implements an **automatic traffic violation detection system** that detects vehicles violating red traffic lights using computer vision. It combines **YOLOv8 object detection**, **SORT tracking**, and **histogram-based red light detection**, and outputs annotated violation clips along with structured JSON records.

---

## 📹 Demo Input

- Sample input video: `vid11_27_7_FaisalTown.mp4`
- Optionally, you can replace it with a real-time **IP webcam stream**.

---

## 📁 Project Structure

```

.
├── inputVideo.mp4                              # Input video or live IP webcam stream
├── violations/violationClips                   # Output violation video clips
├── violations/violation_info                   # Violation metadata in JSON format
├── Histogram/modelTown_histogram.pkl           # Reference histogram for red light detection
└── violationScript.py                          # Detection logic

````

---

## 🚀 How It Works

1. **Red Light Detection**  
   A histogram is calculated from a fixed traffic light ROI and compared with a precomputed reference. If similarity exceeds a threshold → red light is detected.

2. **Vehicle Detection & Tracking**  
   When red light is active:
   - YOLOv8 detects vehicles (specifically cars).
   - SORT tracker assigns consistent IDs to detected cars.

3. **Violation Detection**  
   Cars are monitored across two virtual lines:
   - If a car crosses from the bottom line to the top line during a red light, it is marked as a **violator**.

4. **Clip & JSON Output**  
   - For each red-light phase, a new video clip is recorded.
   - Violators are stored with frame/time info in a JSON file.

---

## 🧠 Tech Stack

- **Python 3**
- **OpenCV**
- **YOLOv8** 
- **SORT** tracking


---


````

Update your signal and path ROIs based on your camera position:

```python
signal_bbox = (0.485, 0.579, 0.010, 0.044)   # Traffic light ROI (normalized)
path_bbox   = (0.901, 0.731, 0.198, 0.199)   # Road region to monitor
```

---

## 📦 Output Example

* **Violation Clip:**
  `violations/violationClips/violation_clip_1.mp4`

* **JSON Metadata:**

  ```json
  {
    "clip_index": 1,
    "violations": [
      {
        "track_id": 2,
        "frame_number": 128,
        "time_seconds": 6.4,
        "bbox": [1245, 565, 1330, 610]
      }
    ]
  }
  ```

---

## ▶️ Run the System

1) Make sure to first clone git repo of Sort tracker:

```bash
https://github.com/abewley/sort.git
```

2) Change the sort.py file: 
from -----> matplotlib.use('TkAgg')
to -------> matplotlib.use('Agg')

3) Add the Code files within Sort repo 

4) Make a virtual environment and then install dependencies:

```bash
pip install opencv-python numpy ultralytics
```

5) Run the detection:

```bash
python violationScript.py
```

---


