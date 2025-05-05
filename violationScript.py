import os
import time
import json
import threading
import queue
from glob import glob

import gradio as gr
import cv2
import numpy as np
import pickle

from ultralytics import YOLO
from sort import *  # Make sure you have the SORT tracker installed: pip install sort-tracker



# ------------------------------------------------------------------------------------
# 1) GLOBALS & DIRECTORIES
# ------------------------------------------------------------------------------------
video_path = "./vid2_Canal.mp4"


IP_CAMERA_URL = video_path  # Replace with your actual IP webcam stream

# IP_CAMERA_URL = "http://<YOUR_PHONE_IP>:8080/video"  # Replace with your actual IP webcam stream
RESULT_DIR = "./violations/violationClips"
JSON_DIR = "./violations/violation_info"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# The detection thread will set this event when told to stop
STOP_EVENT = threading.Event()

# We'll store the latest annotated frame here so Gradio can display it
FRAME_QUEUE = queue.Queue(maxsize=1)

# Flag to indicate if detection is active
IS_RUNNING = False


# ------------------------------------------------------------------------------------
# 2) RED-LIGHT HISTOGRAM & DETECTION FUNCTIONS
# ------------------------------------------------------------------------------------

def load_histogram(file_path):
    """Load a precomputed histogram from a pickle file."""
    with open(file_path, 'rb') as f:
        hist = pickle.load(f)
    return hist

def calculate_histogram(image, bbox):
    
    """Calculate and normalize the histogram for a region in the image."""
    height, width, _ = image.shape
    x_center, y_center, box_width, box_height = bbox

    x_min = int((x_center - box_width / 2) * width)
    x_max = int((x_center + box_width / 2) * width)
    y_min = int((y_center - box_height / 2) * height)
    y_max = int((y_center + box_height / 2) * height)

    roi = image[y_min:y_max, x_min:x_max]
    histogram = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram


def compare_histograms(h1, h2):
    """Compare two histograms using correlation; returns similarity score."""
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

def draw_arrow(frame, x1, y1, x2, y2):
    """Draw a large, filled downward arrow above a bounding box for violation."""
    center_x = (x1 + x2) // 2
    top_y = y1 - 150  # position arrow 150px above the bounding box
    arrow_width = 50
    arrow_height = 100
    shaft_width = 20

    arrow_tip = (center_x, y1)
    left_corner = (center_x - arrow_width, top_y + arrow_height)
    right_corner = (center_x + arrow_width, top_y + arrow_height)
    shaft_top_left = (center_x - shaft_width, top_y)
    shaft_top_right = (center_x + shaft_width, top_y)
    shaft_bottom_left = (center_x - shaft_width, top_y + arrow_height)
    shaft_bottom_right = (center_x + shaft_width, top_y + arrow_height)

    arrow_head = np.array([arrow_tip, left_corner, right_corner], np.int32)
    arrow_shaft = np.array([shaft_top_left, shaft_bottom_left, 
                            shaft_bottom_right, shaft_top_right], np.int32)
    color = (0, 0, 255)
    cv2.fillPoly(frame, [arrow_head], color)
    cv2.fillPoly(frame, [arrow_shaft], color)

def is_within_path(cx, cy, path_bbox, frame_shape):
    """Check if the car center (cx,cy) is within the path region (normalized coords)."""
    h, w, _ = frame_shape
    x_center, y_center, box_w, box_h = path_bbox

    x_min = int((x_center - box_w / 2) * w)
    x_max = int((x_center + box_w / 2) * w)
    y_min = int((y_center - box_h / 2) * h)
    y_max = int((y_center + box_h / 2) * h)
    return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)

def point_side(point, line_start, line_end):
    """Return cross product sign to see if 'point' is left(+) or right(-) of the line."""
    return ((point[0] - line_start[0]) * (line_end[1] - line_start[1]) -
            (point[1] - line_start[1]) * (line_end[0] - line_start[0]))


# ------------------------------------------------------------------------------------
# 3) DETECTION THREAD & MULTI-CLIP LOGIC
# ------------------------------------------------------------------------------------

def detection_loop():
    """
    This function runs in the background. It:
    - Opens the IP camera.
    - Checks for red light via histogram comparison.
    - When red is detected, runs YOLOv8 + SORT on frames to track cars,
      saving separate clips + JSON each time the red period ends.
    - For each new frame, it also posts the annotated frame to FRAME_QUEUE for Gradio display.
    """
    global IS_RUNNING

    # Initialize YOLO, SORT, hist references
    model = YOLO("yolov8n.pt")
    tracker = Sort()
    
    # Load your reference histogram for the red light
    reference_hist = load_histogram("./Histogram/canal_histogram.pkl")


 
    # For Model Town
    # signal_bbox = (0.485, 0.579, 0.010, 0.044)   # normalized ROI for traffic light
    # path_bbox   = (0.901, 0.731, 0.198, 0.199)   # normalized path region
    # lower_line_start = (1332, 1061)  # example pixel coords
    # lower_line_end   = (1912, 842)
    # upper_line_start = (1389, 699)
    # upper_line_end   = (1859, 666)
    # threshold = 0.18
    
    # For Canal
    bbox =(0.246313, 0.375587, 0.041298, 0.154430)  # Normalized coordinates
    path_bbox = (0.720047, 0.827044, 0.559906, 0.345912)  # New specified path coordinate
    lower_line_start = (696,1069)
    lower_line_end   = (1916,1064)
    upper_line_start = (657,953)
    upper_line_end   = (1920,959)
    threshold = 0.18

    cap = cv2.VideoCapture(IP_CAMERA_URL)
    if not cap.isOpened():
        print("[ERROR] Could not open IP camera stream.")
        IS_RUNNING = False
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 20.0

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Multi-clip management
    clip_count = 0
    video_writer = None
    was_red = False
    frame_count = 0

    # Tracking info for the current red phase
    track_states = {}
    clip_violations = {}

    while not STOP_EVENT.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or camera read error.")
            break

        frame_count += 1

        # 1) Red-light check
        hist = calculate_histogram(frame, signal_bbox)
        score = compare_histograms(hist, reference_hist)
        current_red = (score > threshold)

        # 2) If we see a transition from not-red -> red => new clip
        if current_red and not was_red:
            clip_count += 1
            clip_filename = os.path.join(RESULT_DIR, f"violation_clip_{clip_count}.mp4")
            print(f"[INFO] Starting new violation clip: {clip_filename}")
            video_writer = cv2.VideoWriter(clip_filename, fourcc, fps, (frame_width, frame_height))
            track_states = {}
            clip_violations = {}

        # 3) If we see a transition from red -> not red => close the clip, write JSON
        if was_red and not current_red:
            print("[INFO] Stopping current violation clip.")
            if video_writer is not None:
                video_writer.release()
                video_writer = None

            # Save out JSON
            json_filename = os.path.join(JSON_DIR, f"violation_clip_{clip_count}.json")
            violation_data = {
                "clip_index": clip_count,
                "violations": []
            }
            for t_id, info in clip_violations.items():
                violation_data["violations"].append({
                    "track_id": t_id,
                    "frame_number": info["frame_number"],
                    "time_seconds": info["time_seconds"],
                    "bbox": info["bbox"]
                })

            if len(violation_data["violations"]) > 0:
                with open(json_filename, 'w') as f:
                    json.dump(violation_data, f, indent=2)
                print(f"[INFO] JSON saved: {json_filename}")
            else:
                print("[INFO] No violations recorded for this clip.")

        # 4) If currently red, run YOLO detection + SORT
        if current_red:
            results = model(frame)
            detections = []
            for det in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = det[:6]
                if int(cls_id) == 2:  # class 2 = 'car'
                    detections.append([x1, y1, x2, y2, conf])

            detections = np.array(detections) if len(detections) > 0 else np.empty((0,5))
            tracked_objs = tracker.update(detections)

            # For each tracked car, see if it crosses from lower line to upper line => violation
            for x1, y1, x2, y2, track_id in tracked_objs:
                track_id = int(track_id)
                cx = int((x1 + x2)/2)
                cy = int((y1 + y2)/2)

                # Only consider vehicles within path region
                if not is_within_path(cx, cy, path_bbox, frame.shape):
                    continue

                # If new track, check if it's above the lower line
                if track_id not in track_states:
                    if point_side((cx, cy), lower_line_start, lower_line_end) > 0:
                        track_states[track_id] = {"entered": True, "violated": False}
                else:
                    # If it has "entered" and crosses above the upper line => violation
                    if (track_states[track_id]["entered"] and
                        not track_states[track_id]["violated"] and
                        point_side((cx, cy), upper_line_start, upper_line_end) > 0):
                        track_states[track_id]["violated"] = True
                        # Record it in clip_violations
                        clip_violations[track_id] = {
                            "frame_number": frame_count,
                            "time_seconds": round(frame_count/fps, 2),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        }

                # Draw bounding box on the frame
                violated = track_states.get(track_id, {}).get("violated", False)
                color = (0, 0, 255) if violated else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                if violated:
                    draw_arrow(frame, int(x1), int(y1), int(x2), int(y2))

            # Also write the frame to the open clip
            if video_writer is not None:
                video_writer.write(frame)

        # 5) Draw references (lines, signal ROI, score, etc.) for visualization
        h, w, _ = frame.shape
        # Draw lines
        cv2.line(frame, lower_line_start, lower_line_end, (255, 0, 0), 2)    # blue
        cv2.line(frame, upper_line_start, upper_line_end, (0, 255, 255), 2) # yellow

        # Draw signal ROI + score
        sx, sy, sw, sh = signal_bbox
        sx_min = int((sx - sw/2)*w)
        sx_max = int((sx + sw/2)*w)
        sy_min = int((sy - sh/2)*h)
        sy_max = int((sy + sh/2)*h)
        cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {score:.2f}", (sx_min, sy_max + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Path ROI
        px, py, pw, ph = path_bbox
        px_min = int((px - pw/2)*w)
        px_max = int((px + pw/2)*w)
        py_min = int((py - ph/2)*h)
        py_max = int((py + ph/2)*h)
        cv2.rectangle(frame, (px_min, py_min), (px_max, py_max), (0,255,0), 2)

        # Convert BGR -> RGB for Gradio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Post the annotated frame into the queue
        if not FRAME_QUEUE.empty():
            _ = FRAME_QUEUE.get()  # discard older frame if not fetched
        FRAME_QUEUE.put(frame_rgb)

        was_red = current_red

        # Slight pause (this also helps CPU usage)
        time.sleep(0.02)

    # Cleanup if we exit mid-red
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    cap.release()
    IS_RUNNING = False
    print("[INFO] Detection thread ended.")


# ------------------------------------------------------------------------------------
# 4) GRADIO APP SETUP
# ------------------------------------------------------------------------------------

def video_stream_generator():
    """
    Generator for Gradio's live video feed. It yields the latest frame from FRAME_QUEUE.
    """
    while IS_RUNNING:
        if not FRAME_QUEUE.empty():
            frame_rgb = FRAME_QUEUE.get()
            yield frame_rgb
        else:
            time.sleep(0.01)

def start_detection():
    """
    Called by the 'Proceed' button in Gradio. Launches the detection thread if not already running.
    """
    global IS_RUNNING
    if IS_RUNNING:
        return "Detection is already running..."
    IS_RUNNING = True
    STOP_EVENT.clear()
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()
    return "Detection started!"

def stop_detection():
    """
    Called by the 'Stop' button in Gradio. Sets STOP_EVENT, telling detection_loop to exit.
    """
    global IS_RUNNING
    if not IS_RUNNING:
        return "No detection is running."
    STOP_EVENT.set()
    return "Stopping detection..."

def get_violation_videos():
    """
    Return (label, value) pairs for all .mp4 files in RESULT_DIR, sorted by mtime (descending).
    """
    mp4_files = sorted(
        glob(os.path.join(RESULT_DIR, "*.mp4")),
        key=os.path.getmtime,
        reverse=True
    )
    results = []
    for path in mp4_files:
        base = os.path.basename(path)
        results.append((base, path))
    return results

def refresh_video_list():
    """
    Timer callback to update the violation video dropdown with newly found .mp4 files.
    """
    return gr.update(choices=get_violation_videos())

def load_selected_video_info(selected_mp4):
    """
    When a user picks a video from the dropdown, display it in the player and load JSON if available.
    """
    if not selected_mp4 or not os.path.isfile(selected_mp4):
        return (gr.update(value=None), "No video selected or file missing.", "")

    # Attempt to find matching JSON
    base = os.path.basename(selected_mp4)  # e.g. violation_clip_1.mp4
    json_name = os.path.splitext(base)[0] + ".json"  # => violation_clip_1.json
    json_path = os.path.join(JSON_DIR, json_name)

    video_update = gr.update(value=selected_mp4)  # local file path => Gradio will embed it
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        text_info = json.dumps(data, indent=2)
        msg = f"Loaded JSON: {json_name}"
    else:
        text_info = ""
        msg = f"No JSON found for {base}"

    return (video_update, msg, text_info)


# ------------------------------------------------------------------------------------
# 5) BUILD GRADIO INTERFACE
# ------------------------------------------------------------------------------------

with gr.Blocks(title="Red Light Violation Detection") as demo:
    gr.Markdown("## Real-Time Red Light Violation Detection with Multi-Clip Recording + JSON")

    with gr.Row():
        # Left column: Live feed and controls
        with gr.Column():
            live_video = gr.Image(label="Live Feed", show_label=True)
            with gr.Row():
                start_btn = gr.Button("Proceed")
                stop_btn = gr.Button("Stop")
            status_box = gr.Textbox(label="Status", interactive=False)

        # Right column: Violation result videos
        with gr.Column():
            gr.Markdown("### Violation Result Videos")
            violation_list = gr.Dropdown(choices=[], label="Select a Violation Video", value=None)
            load_msg = gr.Text(label="Video / JSON Info")
            with gr.Row():
                selected_video_player = gr.Video(label="Selected Clip", show_label=True, autoplay=False)
                json_display = gr.Code(label="JSON Info", language="json")

    # -- Button actions --
    start_btn.click(fn=start_detection, outputs=status_box)
    stop_btn.click(fn=stop_detection, outputs=status_box)

    # -- Live video feed from detection thread --
    live_video.stream(fn=video_stream_generator)

    # -- Timer to refresh the list of violation videos (every 5s) --
    refresher = gr.Timer(value=5.0)
    refresher.tick(fn=refresh_video_list, outputs=violation_list)


    # -- On user selection => load clip + relevant JSON
    violation_list.change(
        fn=load_selected_video_info,
        inputs=violation_list,
        outputs=[selected_video_player, load_msg, json_display]
    )


# ------------------------------------------------------------------------------------
# 6) RUN THE APP
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    demo.launch(server_name="0.0.0.0", server_port=7850)