import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import sqrt
from deep_sort_realtime.deepsort_tracker import DeepSort


# -----------------------------
# Helper function: compute IoU between two boxes (x1, y1, x2, y2)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


# -----------------------------
# Create folder structure for results
# -----------------------------
base_dir = "results"
video_dir = os.path.join(base_dir, "video")
graph_dir = os.path.join(base_dir, "graphs")
image_dir = os.path.join(base_dir, "images")
data_dir = os.path.join(base_dir, "data")
for folder in [base_dir, video_dir, graph_dir, image_dir, data_dir]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# Setup: Load YOLO model and extract class IDs
# -----------------------------
model = YOLO("yolo12n.pt")  # or "yolov8n.pt"
person_class_id = next((k for k, v in model.names.items() if v == "person"), None)
sports_ball_class_id = next((k for k, v in model.names.items() if v == "sports ball"), None)
if person_class_id is None or sports_ball_class_id is None:
    raise ValueError("Required classes ('person' and 'sports ball') not found in model.names.")
print(f"Person class ID: {person_class_id}")
print(f"Sports ball class ID: {sports_ball_class_id}")

# -----------------------------
# Video input/output setup
# -----------------------------
video_path = "/home/omkedar/PycharmProjects/Major Project/Hand Detection/football-video-analysis/demo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Could not open video at {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------
# Initialize DeepSORT tracker from deep-sort-realtime
# -----------------------------
deepsort = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

# -----------------------------
# Initialize other tracking variables
# -----------------------------
output_frames = []  # Annotated frames for output video
unique_objects = {}  # Unique tracker IDs by class
detections_per_frame = []  # For graphing detections per frame

# Ball tracking variables
ball_positions = []  # (center_x, center_y, frame_number)
total_distance = 0.0
last_ball_center = None
ball_speed = 0.0
ball_acceleration = 0.0

# Possession and pass variables
ball_possession = []  # (frame_number, person_tracker_id)
possession_time = {}  # person_tracker_id -> seconds
passes = []  # (from_id, to_id, start_frame, end_frame)
saved_key_frames = set()
current_possessor = None
possession_frames = 0  # Number of frames the current possessor has held the ball

# Possession lock and smoothing parameters
poss_lock_threshold = 15  # Minimum frames to lock possession before switching
history_length = 5  # Number of frames to average over for the possessor's bounding box
poss_history = []  # List to store possessor bounding boxes over the last few frames

# Player naming for stable identification
player_names = {}  # Mapping from deep-sort tracker id to player name
next_player_num = 1
prev_players = []  # List of tuples: (player_name, center)

# -----------------------------
# Initialize supervision annotators
# -----------------------------
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

display_width = 480
display_height = 270
frame_number = 0

# -----------------------------
# Process video frames
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing complete.")
        break

    # FIX: Always define current_possessor_box here to avoid NameError
    current_possessor_box = None

    # Run YOLO detection on frame
    results = model.predict(frame, conf=0.5)
    try:
        detections = sv.Detections.from_yolov8(results[0])
    except AttributeError:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        detections = sv.Detections(xyxy=boxes, confidence=scores, class_id=class_ids.astype(int))

    # Filter detections for persons and sports ball
    valid_mask = np.isin(detections.class_id, [person_class_id, sports_ball_class_id])
    detections = sv.Detections(
        xyxy=detections.xyxy[valid_mask],
        confidence=detections.confidence[valid_mask],
        class_id=detections.class_id[valid_mask]
    )

    # Prepare detections for DeepSORT: each detection is [ [x1,y1,w,h], conf, class ]
    ds_input = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        conf = float(detections.confidence[i])
        cls = int(detections.class_id[i])
        ds_input.append([[x1, y1, x2 - x1, y2 - y1], conf, cls])
    ds_input = list(ds_input)
    tracks = deepsort.update_tracks(ds_input, frame=frame)

    # Convert DeepSORT output to supervision Detections format
    track_bboxes = []
    track_ids = []
    track_classes = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltwh()  # (x, y, w, h)
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        track_bboxes.append([x1, y1, x2, y2])
        track_ids.append(int(track.track_id))
        track_classes.append(track.det_class)
    if len(track_bboxes) > 0:
        track_bboxes = np.array(track_bboxes)
        track_ids = np.array(track_ids)
        track_classes = np.array(track_classes)
        tracked_detections = sv.Detections(
            xyxy=track_bboxes,
            confidence=np.ones(len(track_bboxes)),
            class_id=track_classes,
            tracker_id=track_ids
        )
    else:
        tracked_detections = sv.Detections.empty()

    # Ensure tracker_ids and class_ids are iterable
    if tracked_detections.tracker_id is None:
        tracker_ids = []
        class_ids = []
    else:
        tracker_ids = tracked_detections.tracker_id
        class_ids = tracked_detections.class_id

    # Update unique objects summary
    if len(tracker_ids) > 0:
        for cls_id, t_id in zip(tracker_ids, class_ids):
            cls_id = int(cls_id)
            t_id = int(t_id)
            if cls_id not in unique_objects:
                unique_objects[cls_id] = set()
            unique_objects[cls_id].add(t_id)
    detections_per_frame.append(len(tracked_detections))

    # -----------------------------
    # Ball tracking and possession detection
    # -----------------------------
    ball_center = None
    ball_mask = (tracked_detections.class_id == sports_ball_class_id)
    if np.any(ball_mask):
        ball_box = tracked_detections.xyxy[ball_mask][0]
        bx1, by1, bx2, by2 = map(float, ball_box)
        bx = (bx1 + bx2) / 2
        by = (by1 + by2) / 2
        ball_center = (bx, by)
        ball_positions.append((bx, by, frame_number))
        if last_ball_center is not None:
            dist = sqrt((bx - last_ball_center[0]) ** 2 + (by - last_ball_center[1]) ** 2)
            total_distance += dist
            new_speed = dist * fps
            ball_acceleration = (new_speed - ball_speed) * fps
            ball_speed = new_speed
        last_ball_center = ball_center

        # Possession detection: choose the person enclosing the ball or closest (within 150 px)
        person_mask = (tracked_detections.class_id == person_class_id)
        best_id = None
        min_dist = 1e9
        best_box = None
        if np.any(person_mask):
            p_boxes = tracked_detections.xyxy[person_mask]
            p_ids = tracked_detections.tracker_id[person_mask]
            for p_box, pid in zip(p_boxes, p_ids):
                px1, py1, px2, py2 = map(float, p_box)
                # Check if ball center is within the box
                if bx >= px1 and bx <= px2 and by >= py1 and by <= py2:
                    best_id = int(pid)
                    min_dist = 0
                    best_box = p_box
                    break
                else:
                    pcx = (px1 + px2) / 2
                    pcy = (py1 + py2) / 2
                    d = sqrt((pcx - bx) ** 2 + (pcy - by) ** 2)
                    if d < min_dist and d < 150:
                        min_dist = d
                        best_id = int(pid)
                        best_box = p_box
        # Extra temporal consistency: if previous possessor is nearly as close (within 20% margin), keep it.
        if best_id is not None and current_possessor is not None and best_id != current_possessor:
            poss_mask = (tracked_detections.tracker_id == current_possessor)
            if np.any(poss_mask):
                p_box = tracked_detections.xyxy[poss_mask][0]
                px1, py1, px2, py2 = map(float, p_box)
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                poss_dist = sqrt((pcx - bx) ** 2 + (pcy - by) ** 2)
                if poss_dist < min_dist * 1.2:
                    best_id = current_possessor
                    best_box = p_box

        # Possession lock with temporal smoothing:
        if best_id is not None:
            if current_possessor is None or best_id == current_possessor:
                possession_frames += 1
                poss_history.append(best_box)
                if len(poss_history) > history_length:
                    poss_history = poss_history[-history_length:]
            else:
                if current_possessor is not None and possession_frames < poss_lock_threshold:
                    best_id = current_possessor
                    if poss_history:
                        best_box = np.mean(poss_history, axis=0)
                else:
                    possession_frames = 1
                    poss_history = [best_box]
                    if current_possessor is not None and current_possessor != best_id:
                        passes.append((current_possessor, best_id, frame_number - 1, frame_number))
                    current_possessor = best_id
            ball_possession.append((frame_number, best_id))
            possession_time[best_id] = possession_time.get(best_id, 0) + (1 / fps)
            current_possessor_box = np.mean(poss_history, axis=0)
        else:
            current_possessor_box = None
    else:
        if current_possessor is not None:
            ball_possession.append((frame_number, current_possessor))
            current_possessor_box = None

    # -----------------------------
    # Annotate current frame
    # -----------------------------
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
    base_labels = [f"ID: {int(t_id)} {model.names[int(cls_id)]}"
                   for t_id, cls_id in zip(tracker_ids, class_ids)]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=base_labels)

    # --- Add individual player names and extra info ---
    current_frame_players = []
    if len(tracker_ids) > 0:
        for det, t_id, cls_id in zip(tracked_detections.xyxy, tracker_ids, class_ids):
            if int(cls_id) == person_class_id:
                pid = int(t_id)
                x1, y1, x2, y2 = map(int, det)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                if pid in player_names:
                    assigned_name = player_names[pid]
                else:
                    assigned_name = None
                    for prev_name, prev_center in prev_players:
                        if sqrt((prev_center[0] - center[0]) ** 2 + (prev_center[1] - center[1]) ** 2) < 50:
                            assigned_name = prev_name
                            break
                    if assigned_name is None:
                        assigned_name = f"Player {next_player_num}"
                        next_player_num += 1
                    player_names[pid] = assigned_name
                current_frame_players.append((assigned_name, center))
                cv2.putText(annotated_frame, assigned_name, (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if ball_center is not None:
                    d = sqrt((center[0] - ball_center[0]) ** 2 + (center[1] - ball_center[1]) ** 2)
                    if d < 300 and pid != current_possessor:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(annotated_frame, "Nearby", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    prev_players = current_frame_players.copy()

    # Highlight the possessor with a red box
    if current_possessor_box is not None:
        px1, py1, px2, py2 = map(int, current_possessor_box)
        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(annotated_frame, "Possessor", (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw ball trajectory with trailing effect (last 3 seconds)
    trail_frames = int(fps * 3)
    min_trail_frame = frame_number - trail_frames
    recent_positions = [p for p in ball_positions if p[2] >= min_trail_frame]
    for i in range(1, len(recent_positions)):
        x1, y1, _ = recent_positions[i - 1]
        x2, y2, _ = recent_positions[i]
        cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # -----------------------------
    # Stats Display Fix
    # -----------------------------
    if current_possessor is not None:
        # If we have a current possessor, use them
        poss_name = player_names.get(current_possessor, f"ID {current_possessor}")
    elif ball_possession:
        # Otherwise, if we have recorded possessions, use the last one
        last_possessor_id = ball_possession[-1][1]
        poss_name = player_names.get(last_possessor_id, f"ID {last_possessor_id}")
    else:
        # Otherwise, none
        poss_name = "None"

    stats_text = (
        f"Frame: {frame_number} | Objects: {len(tracked_detections)}\n"
        f"Ball Dist: {total_distance:.2f} px | Speed: {ball_speed:.2f} px/s | Accel: {ball_acceleration:.2f} px/s²\n"
        f"Possession: {poss_name}"
    )
    for i, line in enumerate(stats_text.split("\n")):
        cv2.putText(annotated_frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    disp_frame = cv2.resize(annotated_frame, (display_width, display_height))
    cv2.imshow("Football Match Detection", disp_frame)
    output_frames.append(annotated_frame.copy())

    if len(ball_possession) >= 2:
        if ball_possession[-1][1] != ball_possession[-2][1]:
            pass_event = (ball_possession[-2][1], ball_possession[-1][1],
                          ball_possession[-2][0], ball_possession[-1][0])
            passes.append(pass_event)
            key_frame_id = ball_possession[-1][0]
            if key_frame_id not in saved_key_frames:
                if current_possessor_box is not None:
                    cx_current = (current_possessor_box[0] + current_possessor_box[2]) / 2
                    cy_current = (current_possessor_box[1] + current_possessor_box[3]) / 2
                else:
                    cx_current, cy_current = 0, 0
                cv2.imwrite(os.path.join(image_dir, f"key_frame_{key_frame_id}.png"), annotated_frame)
                prev_keyframe_box = (cx_current, cy_current)
                saved_key_frames.add(key_frame_id)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Save annotated video
# -----------------------------
output_video_path = os.path.join(video_dir, "annotated_football_match.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
for frame in output_frames:
    out.write(frame)
out.release()
print(f"Annotated video saved as {output_video_path}")

# -----------------------------
# Save CSV files
# -----------------------------
ball_df = pd.DataFrame(ball_positions, columns=["center_x", "center_y", "frame"])
ball_df.to_csv(os.path.join(data_dir, "ball_trajectory.csv"), index=False)

possession_df = pd.DataFrame(ball_possession, columns=["frame", "person_tracker_id"])
possession_df["player_name"] = possession_df["person_tracker_id"].apply(lambda x: player_names.get(x, f"ID_{x}"))
possession_df.to_csv(os.path.join(data_dir, "ball_possession.csv"), index=False)

passes_df = pd.DataFrame(passes, columns=["from_person_id", "to_person_id", "start_frame", "end_frame"])
passes_df["from_player_name"] = passes_df["from_person_id"].apply(lambda x: player_names.get(x, f"ID_{x}"))
passes_df["to_player_name"] = passes_df["to_person_id"].apply(lambda x: player_names.get(x, f"ID_{x}"))
passes_df.to_csv(os.path.join(data_dir, "pass_events.csv"), index=False)

print("Detection summary (unique objects):")
for cls_id, tracker_ids in unique_objects.items():
    class_name = model.names[cls_id]
    unique_count = len(tracker_ids)
    print(f"{class_name}: {unique_count}")

print("\nPass Summary:")
for pass_event in passes:
    from_id, to_id, start_frame, end_frame = pass_event
    print(f"Ball passed from ID {from_id} to ID {to_id} between frames {start_frame} and {end_frame}")

print("\nPossession Time (in seconds):")
for person_id, time_sec in possession_time.items():
    print(f"Player {player_names.get(person_id, f'ID_{person_id}')}: {time_sec:.2f} sec")

summary_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
for i in range(1, len(ball_positions)):
    x1, y1, _ = ball_positions[i - 1]
    x2, y2, _ = ball_positions[i]
    cv2.line(summary_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
for pass_event in passes:
    from_id, to_id, start_frame, end_frame = pass_event
    start_pos = next((pos for pos in ball_positions if pos[2] == start_frame), None)
    end_pos = next((pos for pos in ball_positions if pos[2] == end_frame), None)
    if start_pos and end_pos:
        x1, y1, _ = start_pos
        x2, y2, _ = end_pos
        cv2.circle(summary_frame, (int(x1), int(y1)), 8, (0, 255, 0), -1)
        cv2.circle(summary_frame, (int(x2), int(y2)), 8, (0, 0, 255), -1)
        from_name = player_names.get(from_id, f"ID_{from_id}")
        to_name = player_names.get(to_id, f"ID_{to_id}")
        label = f"{from_name} → {to_name}"
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        cv2.putText(summary_frame, label, (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imwrite(os.path.join(image_dir, "ball_trajectory_and_passes.png"), summary_frame)
print("Summary image saved as ball_trajectory_and_passes.png")

plt.figure(figsize=(10, 5))
plt.plot(detections_per_frame, label="Objects Detected per Frame")
plt.xlabel("Frame Number")
plt.ylabel("Number of Objects")
plt.title("Detections Over Time")
plt.legend()
plt.savefig(os.path.join(graph_dir, "detections_over_time.png"))
plt.close()
print("Detection graph saved as detections_over_time.png")

distances = []
cumulative_distance = 0
for i in range(1, len(ball_positions)):
    x1, y1, _ = ball_positions[i - 1]
    x2, y2, _ = ball_positions[i]
    d = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    cumulative_distance += d
    distances.append(cumulative_distance)
plt.figure(figsize=(10, 5))
plt.plot(distances, label="Cumulative Ball Distance (px)")
plt.xlabel("Frame Number")
plt.ylabel("Distance (pixels)")
plt.title("Ball Distance Over Time")
plt.legend()
plt.savefig(os.path.join(graph_dir, "ball_distance_over_time.png"))
plt.close()
print("Ball distance graph saved as ball_distance_over_time.png")
