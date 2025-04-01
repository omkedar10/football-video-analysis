import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import sqrt
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from collections import deque

# --- MiDaS Setup ---
print("Loading MiDaS model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
except Exception as e: print(f"Fatal Error loading MiDaS model: {e}"); exit()
midas.to(device); midas.eval(); print("MiDaS model loaded.")

def estimate_depth(image):
    if image is None: return None
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = transform(rgb).to(device)
        if input_batch.dim() == 3: input_batch = input_batch.unsqueeze(0)
        with torch.no_grad(): depth_prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate( depth_prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False, ).squeeze().cpu().numpy()
        return depth_map
    except Exception as e: print(f"Error: Depth estimation failed: {e}"); return None

# Using minimum depth in patch - keep this attempt for now
def get_minimum_depth_in_patch(depth_map, box, patch_scale=0.6):
    if depth_map is None or box is None: return None
    h_map, w_map = depth_map.shape; x1, y1, x2, y2 = map(int, box)
    if x1 >= x2 or y1 >= y2: return None
    x1_c, x2_c = max(0, x1), min(w_map - 1, x2); y1_c, y2_c = max(0, y1), min(h_map - 1, y2)
    if x1_c >= x2_c or y1_c >= y2_c: return None
    box_w = x2_c - x1_c; box_h = y2_c - y1_c
    if box_w < 5 or box_h < 5: return None
    patch_w = int(box_w * patch_scale); patch_h = int(box_h * patch_scale)
    patch_x1 = x1_c + (box_w - patch_w) // 2; patch_y1 = y1_c + (box_h - patch_h) // 2
    patch_x2 = patch_x1 + patch_w; patch_y2 = patch_y1 + patch_h
    patch_x1, patch_x2 = max(0, patch_x1), min(w_map - 1, patch_x2); patch_y1, patch_y2 = max(0, patch_y1), min(h_map - 1, patch_y2)
    if patch_x1 >= patch_x2 or patch_y1 >= patch_y2: return None
    roi = depth_map[patch_y1:patch_y2, patch_x1:patch_x2]
    if roi.size == 0: return None
    valid_roi = roi[roi > 1e-6]
    if valid_roi.size < 5: return None
    min_depth = np.min(valid_roi)
    if np.isnan(min_depth): return None
    return min_depth

# --- Folder Structure ---
base_dir = "results_depth_highlight";
video_dir=os.path.join(base_dir, "video"); graph_dir=os.path.join(base_dir, "graphs"); image_dir=os.path.join(base_dir, "images"); data_dir=os.path.join(base_dir, "data"); depth_dir=os.path.join(base_dir, "depth_maps")
for folder in [base_dir, video_dir, graph_dir, image_dir, data_dir, depth_dir]: os.makedirs(folder, exist_ok=True)

# --- YOLO Setup ---
print("Loading YOLO model (yolo12n.pt)..."); model_path = "yolo12n.pt"
if not os.path.exists(model_path): raise FileNotFoundError(f"YOLO model not found: {model_path}")
model = YOLO(model_path)
person_class_id = next((k for k, v in model.names.items() if v == "person"), None)
sports_ball_class_id = next((k for k, v in model.names.items() if v == "sports ball"), None)
if person_class_id is None or sports_ball_class_id is None: raise ValueError(f"Classes not found in {model_path}")
print(f"Person ID: {person_class_id}, Ball ID: {sports_ball_class_id}. YOLO loaded.")

# --- Video I/O ---
video_path = "/home/omkedar/PycharmProjects/Major Project/Hand Detection/football-video-analysis/demo.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): raise ValueError(f"Cannot open video: {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS); frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video loaded: {video_path} ({frame_width}x{frame_height} @ {fps:.2f} FPS)")

# --- Tracker & Variables Init ---
print("Initializing DeepSORT..."); deepsort = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7); print("DeepSORT initialized.")
output_frames=[]; unique_objects={}; detections_per_frame=[]
ball_positions=[]; total_distance=0.0; last_ball_center=None; ball_speed=0.0; ball_acceleration=0.0
ball_possession=[]; possession_time={}; passes=[]; saved_key_frames=set()
current_possessor_id = None; current_possessor_smoothed_box = None; possession_frames = 0 # Start as None
poss_lock_threshold = 10; history_length = 5; poss_history_boxes = deque(maxlen=history_length)
player_names={}; next_player_num=1; prev_players=[]
depth_history_len=5; ball_depth_history={}; player_depth_history={}
depth_info_rows=[]
box_annotator=sv.BoxAnnotator(thickness=2); label_annotator=sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_padding=2)
display_scale=0.5; display_width=int(frame_width*display_scale); display_height=int(frame_height*display_scale)
frame_number = 0
print("Starting video processing loop...")

# ================================================
# Main Loop
# ================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: print("Video processing complete."); break

    annotated_frame = frame.copy()

    # --- Depth Map ---
    depth_map = estimate_depth(frame)
    if depth_map is None: print(f"Warn: Skip frame {frame_number}, depth error."); frame_number+=1; continue
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    depth_annotated_current_frame = depth_colored.copy()

    # --- Reset Vars ---
    current_possessor_smoothed_box = None # Will be recalc

    # --- YOLO Detection ---
    results = model.predict(frame, conf=0.5, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    valid_mask = np.isin(detections.class_id, [person_class_id, sports_ball_class_id])
    detections = detections[valid_mask]

    # --- DeepSORT ---
    ds_input = [[[int(x1),int(y1),int(x2-x1),int(y2-y1)], float(conf), int(cls)] for (x1,y1,x2,y2),conf,cls in zip(detections.xyxy, detections.confidence, detections.class_id)]
    tracks = deepsort.update_tracks(ds_input, frame=frame)
    track_bboxes = []; track_ids = []; track_classes = []
    for track in tracks:
        if not track.is_confirmed(): continue
        bbox_ltwh = track.to_ltwh();
        if bbox_ltwh is None: continue
        x1, y1, w, h = map(int, bbox_ltwh)
        track_bboxes.append([x1, y1, x1 + w, y1 + h])
        track_ids.append(int(track.track_id))
        track_classes.append(track.det_class)

    if len(track_bboxes) > 0:
        tracked_detections = sv.Detections(xyxy=np.array(track_bboxes), class_id=np.array(track_classes), tracker_id=np.array(track_ids))
    else: tracked_detections = sv.Detections.empty()

    if tracked_detections.tracker_id is not None:
        for cls_id, t_id in zip(tracked_detections.class_id, tracked_detections.tracker_id):
            if cls_id not in unique_objects: unique_objects[cls_id] = set()
            unique_objects[cls_id].add(t_id)
    detections_per_frame.append(len(tracked_detections))

    # --- Ball Tracking & Possession ---
    ball_center = None; ball_box = None; ball_track_id = None
    ball_mask = np.array([])
    if tracked_detections.class_id is not None: ball_mask = (tracked_detections.class_id == sports_ball_class_id)
    potential_possessor_raw_box_this_frame = None
    pass_detected_this_frame = False
    potential_possessor_id = None

    if np.any(ball_mask):
        ball_index = np.where(ball_mask)[0][0]
        ball_box = tracked_detections.xyxy[ball_index]
        if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > ball_index: ball_track_id = tracked_detections.tracker_id[ball_index]
        bx1, by1, bx2, by2 = ball_box; ball_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
        ball_positions.append((ball_center[0], ball_center[1], frame_number))

        if last_ball_center is not None:
            dist = sqrt((ball_center[0]-last_ball_center[0])**2 + (ball_center[1]-last_ball_center[1])**2)
            total_distance += dist; new_speed = dist * fps
            ball_acceleration = (new_speed - ball_speed) * fps if ball_speed > 1e-6 else 0
            ball_speed = new_speed
        last_ball_center = ball_center

        # --- Find Closest Player within Increased Threshold ---
        min_dist_to_ball = 250 # <<< INCREASED THRESHOLD
        person_mask = np.array([])
        if tracked_detections.class_id is not None: person_mask = (tracked_detections.class_id == person_class_id)

        if np.any(person_mask) and tracked_detections.tracker_id is not None:
            p_indices = np.where(person_mask)[0]
            # print(f"Frame {frame_number}: Checking {len(p_indices)} persons...") # DEBUG
            for idx in p_indices:
                if idx < len(tracked_detections.tracker_id):
                    p_box = tracked_detections.xyxy[idx]; pid = tracked_detections.tracker_id[idx]
                    pcx = (p_box[0] + p_box[2]) / 2; pcy = (p_box[1] + p_box[3]) / 2
                    dist_to_ball = sqrt((pcx - ball_center[0])**2 + (pcy - ball_center[1])**2)
                    # print(f"  > Person ID {pid} dist: {dist_to_ball:.1f}") # DEBUG Distance
                    if dist_to_ball < min_dist_to_ball:
                        min_dist_to_ball = dist_to_ball
                        potential_possessor_id = int(pid)
                        potential_possessor_raw_box_this_frame = p_box
            # if potential_possessor_id is not None: print(f"  >> Found Potential Possessor: {potential_possessor_id} at {min_dist_to_ball:.1f}") # DEBUG Found

        # --- Possession Stability (Crucial Fix Here) ---
        if potential_possessor_id is not None: # A nearby player was found
            if current_possessor_id is None: # First time OR regaining after loss
                current_possessor_id = potential_possessor_id
                possession_frames = 1
                poss_history_boxes.clear()
                if potential_possessor_raw_box_this_frame is not None: poss_history_boxes.append(potential_possessor_raw_box_this_frame)
            elif potential_possessor_id == current_possessor_id: # Same player continues
                possession_frames += 1
                if potential_possessor_raw_box_this_frame is not None: poss_history_boxes.append(potential_possessor_raw_box_this_frame)
            else: # Different player is now closest
                if possession_frames >= poss_lock_threshold: # Pass occurred
                    passes.append((current_possessor_id, potential_possessor_id, frame_number - possession_frames, frame_number))
                    pass_detected_this_frame = True
                    current_possessor_id = potential_possessor_id
                    possession_frames = 1
                    poss_history_boxes.clear()
                    if potential_possessor_raw_box_this_frame is not None: poss_history_boxes.append(potential_possessor_raw_box_this_frame)
                else: # Debounce - stick with old ID, but smooth box towards new position
                    if potential_possessor_raw_box_this_frame is not None: poss_history_boxes.append(potential_possessor_raw_box_this_frame)

        # --- Critical Change: DO NOT clear current_possessor_id if NO potential is found ---
        # else: # No potential possessor found *nearby* THIS frame
              # DON'T DO THIS -> current_possessor_id = None
              # Keep the existing current_possessor_id unless a pass happens or the track is lost by DeepSORT
              # possession_frames = 0 # Reset frame count if proximity lost, but keep ID

        # Log possession only if an ID is assigned
        if current_possessor_id is not None:
            ball_possession.append((frame_number, current_possessor_id))
            possession_time[current_possessor_id] = possession_time.get(current_possessor_id, 0) + (1 / fps)

        # Update smoothed box if history exists
        if poss_history_boxes: current_possessor_smoothed_box = np.mean(list(poss_history_boxes), axis=0)
        # --- END Possession Stability ---

    else: # Ball not detected
        ball_speed=0.0; ball_acceleration=0.0; last_ball_center=None
        # Clear possessor info if ball is lost
        current_possessor_id = None
        possession_frames = 0
        poss_history_boxes.clear() # Clear box history too

    # --- Depth Calculation ---
    poss_depth_raw=None; poss_depth_smoothed=None; ball_depth_raw=None; ball_depth_smoothed=None

    # Player Depth (minimum in patch)
    if current_possessor_smoothed_box is not None and current_possessor_id is not None:
        poss_depth_raw = get_minimum_depth_in_patch(depth_map, current_possessor_smoothed_box, patch_scale=0.6)
        if poss_depth_raw is not None:
            if current_possessor_id not in player_depth_history: player_depth_history[current_possessor_id] = deque(maxlen=depth_history_len)
            player_depth_history[current_possessor_id].append(poss_depth_raw)
            # Ensure history is not empty before calculating mean
            if player_depth_history[current_possessor_id]:
                 poss_depth_smoothed = np.mean(list(player_depth_history[current_possessor_id]))

    # Ball Depth (minimum in patch)
    if ball_box is not None and ball_track_id is not None:
        ball_depth_raw = get_minimum_depth_in_patch(depth_map, ball_box, patch_scale=0.8)
        if ball_depth_raw is not None:
            if ball_track_id not in ball_depth_history: ball_depth_history[ball_track_id] = deque(maxlen=depth_history_len)
            ball_depth_history[ball_track_id].append(ball_depth_raw)
            if ball_depth_history[ball_track_id]:
                ball_depth_smoothed = np.mean(list(ball_depth_history[ball_track_id]))

    # Combined Metrics
    depth_diff=np.nan; depth_avg=np.nan; combined_text="Combined: N/A"
    # Ensure both smoothed values are valid numbers before calculating
    if ball_depth_smoothed is not None and poss_depth_smoothed is not None and np.isfinite(ball_depth_smoothed) and np.isfinite(poss_depth_smoothed):
        depth_diff = abs(ball_depth_smoothed - poss_depth_smoothed)
        depth_avg = (ball_depth_smoothed + poss_depth_smoothed) / 2
        combined_text = f"Combined: Diff {depth_diff:.2f}, Avg {depth_avg:.2f}"

    # Log Depth
    depth_info_rows.append({ "frame": frame_number, "ball_d_raw": ball_depth_raw, "ball_d_smooth": ball_depth_smoothed,
                             "poss_d_raw": poss_depth_raw, "poss_d_smooth": poss_depth_smoothed,
                             "d_diff_smooth": depth_diff, "d_avg_smooth": depth_avg })

    # Text for Display
    ball_depth_text = f"Ball Depth: {ball_depth_smoothed:.2f}" if ball_depth_smoothed is not None else "Ball Depth: N/A"
    poss_depth_text = f"Poss Depth: {poss_depth_smoothed:.2f}" if poss_depth_smoothed is not None else "Poss Depth: N/A"

    # --- Annotate Main Frame ---
    # Default boxes/labels
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
    current_frame_players=[]; labels_to_draw=[]
    # ...(Player name assignment unchanged)...
    if tracked_detections.tracker_id is not None:
        for i in range(len(tracked_detections)):
            t_id=tracked_detections.tracker_id[i]; cls_id=tracked_detections.class_id[i]; box=tracked_detections.xyxy[i]
            base_label = f"ID:{t_id} {model.names[cls_id]}"
            if cls_id == person_class_id:
                pid=t_id; center=((box[0]+box[2])/2, (box[1]+box[3])/2)
                assigned_name=player_names.get(pid)
                if assigned_name is None:
                    min_dist=75; match_found=False
                    for prev_name, prev_center in prev_players:
                        if sqrt((prev_center[0]-center[0])**2 + (prev_center[1]-center[1])**2) < min_dist:
                            name_used = any(p_name == prev_name and p_id != pid and tracked_detections.tracker_id is not None and p_id in tracked_detections.tracker_id for p_id, p_name in player_names.items() if p_id in tracked_detections.tracker_id)
                            if not name_used: assigned_name=prev_name; match_found=True; break
                    if not match_found: assigned_name = f"Player {next_player_num}"; next_player_num += 1
                    player_names[pid] = assigned_name
                current_frame_players.append((assigned_name, center)); base_label = f"{assigned_name}" # Simpler label
            labels_to_draw.append(base_label)
    if tracked_detections.xyxy.shape[0] > 0 and len(labels_to_draw) == len(tracked_detections.xyxy):
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels_to_draw)
    prev_players = current_frame_players.copy()

    # --- Draw Highlights ---
    # Iterate through current tracked persons
    if tracked_detections.tracker_id is not None:
        person_indices_now = np.where(tracked_detections.class_id == person_class_id)[0]
        for i in person_indices_now:
            if i < len(tracked_detections.tracker_id): # Check index validity
                pid = tracked_detections.tracker_id[i]; box = tracked_detections.xyxy[i]
                x1, y1, x2, y2 = map(int, box); center = ((x1+x2)/2, (y1+y2)/2)

                # Possessor Highlight - Check if current_possessor_id is not None *before* comparing
                if current_possessor_id is not None and pid == current_possessor_id:
                    # print(f"DEBUG Frame {frame_number}: Drawing Possessor Highlight ID: {pid}") # DEBUG
                    p_box_draw = current_possessor_smoothed_box if current_possessor_smoothed_box is not None else box
                    px1_draw, py1_draw, px2_draw, py2_draw = map(int, p_box_draw)
                    cv2.rectangle(annotated_frame, (px1_draw, py1_draw), (px2_draw, py2_draw), (0, 0, 255), 3) # RED
                    cv2.putText(annotated_frame, "Possessor", (px1_draw, py1_draw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Nearby Player Highlight - Check ball_center and ensure not the possessor
                elif ball_center is not None and (current_possessor_id is None or pid != current_possessor_id):
                     dist_to_ball = sqrt((center[0]-ball_center[0])**2 + (center[1]-ball_center[1])**2)
                     if dist_to_ball < 150: # Nearby threshold
                         # print(f"DEBUG Frame {frame_number}: Drawing Nearby Highlight ID: {pid}") # DEBUG
                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 3) # BLUE

    # --- Ball Trajectory ---
    # ...(Unchanged)...
    trail_frames = int(fps * 1.5); min_trail_frame = max(0, frame_number - trail_frames)
    recent_positions = [(p[0],p[1]) for p in ball_positions if p[2] >= min_trail_frame]
    if len(recent_positions) > 5:
        positions_array = np.array(recent_positions); kernel_size = 5; kernel = np.ones(kernel_size) / kernel_size
        smoothed_x = np.convolve(positions_array[:, 0], kernel, mode='valid'); smoothed_y = np.convolve(positions_array[:, 1], kernel, mode='valid')
        smoothed_positions = np.column_stack((smoothed_x, smoothed_y)).astype(int)
        for i in range(1, len(smoothed_positions)): cv2.line(annotated_frame, tuple(smoothed_positions[i-1]), tuple(smoothed_positions[i]), (0, 255, 255), 2)
    elif len(recent_positions) > 1:
        for i in range(1, len(recent_positions)): cv2.line(annotated_frame, (int(recent_positions[i-1][0]), int(recent_positions[i-1][1])), (int(recent_positions[i][0]), int(recent_positions[i][1])), (0, 255, 255), 2)

    # --- Stats Display ---
    # Determine possessor name based on the *final* current_possessor_id for the frame
    if current_possessor_id is not None: poss_name = player_names.get(current_possessor_id, f"ID {current_possessor_id}")
    # If ID is None now, check if it was assigned earlier in the frame via ball_possession log
    elif ball_possession and ball_possession[-1][0] == frame_number: # Check if logged this frame
        last_possessor_id = ball_possession[-1][1]
        poss_name = player_names.get(last_possessor_id, f"ID {last_possessor_id}")
    else: poss_name = "None" # Truly no possessor determined

    stats_lines = [ f"Frame: {frame_number} | Objects: {len(tracked_detections)}",
                    f"Ball Speed: {ball_speed:.1f}px/s | Accel: {ball_acceleration:.1f}px/s²",
                    f"Possession: {poss_name}", # REFLECTS FINAL ID
                    ball_depth_text, poss_depth_text, combined_text ]
    y_offset = 30
    # ...(Drawing text unchanged)...
    for i, line in enumerate(stats_lines):
        cv2.putText(annotated_frame, line, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, line, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # --- Annotate Depth Map ---
    # ...(Annotations depend on valid smoothed depths - unchanged)...
    # Ball
    if ball_box is not None and ball_depth_smoothed is not None:
        bx1, by1, bx2, by2 = map(int, ball_box)
        cv2.rectangle(depth_annotated_current_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(depth_annotated_current_frame, ball_depth_text, (bx1, max(by1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(depth_annotated_current_frame, ball_depth_text, (bx1, max(by1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Possessor
    if current_possessor_id is not None and current_possessor_smoothed_box is not None and poss_depth_smoothed is not None:
        px1, py1, px2, py2 = map(int, current_possessor_smoothed_box)
        cv2.rectangle(depth_annotated_current_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(depth_annotated_current_frame, poss_depth_text, (px1, max(py1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(depth_annotated_current_frame, poss_depth_text, (px1, max(py1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        # No patch coords to draw with min_depth function

    # Combined Text
    cv2.putText(depth_annotated_current_frame, combined_text, (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(depth_annotated_current_frame, combined_text, (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(depth_dir, f"depth_frame_{frame_number:04d}.png"), depth_annotated_current_frame)


    # --- Display & Output ---
    disp_frame = cv2.resize(annotated_frame, (display_width, display_height))
    cv2.imshow("Football Analysis (Depth + Highlights)", disp_frame)
    output_frames.append(annotated_frame)

    # --- Keyframe Saving ---
    if pass_detected_this_frame:
        key_frame_id = frame_number
        if key_frame_id not in saved_key_frames:
             cv2.imwrite(os.path.join(image_dir, f"key_frame_{key_frame_id:04d}.png"), annotated_frame)
             saved_key_frames.add(key_frame_id)

    # --- Exit/Frame Increment ---
    if cv2.waitKey(1) & 0xFF == ord('q'): print("Processing interrupted."); break
    if frame_number > 0 and frame_number % 50 == 0:
        print(f"Processed frame {frame_number}... Current Possessor ID: {current_possessor_id}") # Print current ID
    frame_number += 1

# ...(Cleanup, Saving Video, DataFrames, Summaries, Graphs remain the same as v6)...
cap.release(); cv2.destroyAllWindows(); print("Video capture released.")
print("Saving annotated video..."); output_video_path = os.path.join(video_dir, "annotated_match_v7_final_fix.mp4") # v7
fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
if not out.isOpened(): print(f"Error opening video writer: {output_video_path}")
else:
    for frame_out in output_frames: out.write(frame_out)
    out.release(); print(f"Annotated video saved: {output_video_path}")
print("Saving data files...")
pd.DataFrame(ball_positions, columns=["x", "y", "frame"]).to_csv(os.path.join(data_dir, "ball_trajectory.csv"), index=False)
possession_df = pd.DataFrame(ball_possession, columns=["frame", "id"]); possession_df["player"] = possession_df["id"].apply(lambda x: player_names.get(x, f"ID_{x}"))
possession_df.to_csv(os.path.join(data_dir, "ball_possession.csv"), index=False)
passes_df = pd.DataFrame(passes, columns=["from_id", "to_id", "start_f", "end_f"]); passes_df["from_player"] = passes_df["from_id"].apply(lambda x: player_names.get(x, f"ID_{x}"))
passes_df["to_player"] = passes_df["to_id"].apply(lambda x: player_names.get(x, f"ID_{x}")); passes_df.to_csv(os.path.join(data_dir, "pass_events.csv"), index=False)
depth_df = pd.DataFrame(depth_info_rows); depth_df.columns = ["frame", "ball_d_raw", "ball_d_smooth", "poss_d_raw", "poss_d_smooth", "d_diff_smooth", "d_avg_smooth"]
depth_df.to_csv(os.path.join(data_dir, "depth_info_detailed.csv"), index=False, float_format='%.4f'); print("Data files saved.")
print("\n--- Analysis Summary ---")
print("Unique objects tracked:"); total_obj = 0
for cls_id, tracker_ids in unique_objects.items(): class_name = model.names.get(cls_id, f"?"); count = len(tracker_ids); print(f"- {class_name}: {count}"); total_obj += count
print(f"Total unique objects: {total_obj}")
print("\nPass Summary:");
if passes: [print(f"- {player_names.get(f_id, '?')}->{player_names.get(t_id, '?')} (f{s_f}-{e_f})") for f_id, t_id, s_f, e_f in passes]
else: print("- None")
print("\nPossession Time (s):")
if possession_time: [print(f"- {player_names.get(p_id, '?')}: {t:.2f}s") for p_id, t in sorted(possession_time.items(), key=lambda item: item[1], reverse=True)]
else: print("- None")
print("\nGenerating graphs...")
plt.figure(figsize=(10, 5)); plt.plot(range(frame_number), detections_per_frame, label="# Objects/Frame"); plt.xlabel("Frame"); plt.ylabel("Count"); plt.title("Tracked Objects"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(graph_dir, "detections.png")); plt.close(); print("- Detections graph saved.")
if ball_positions:
    dists = [0]; cum_d = 0
    for i in range(1, len(ball_positions)): x1,y1,_=ball_positions[i-1]; x2,y2,_=ball_positions[i]; d=sqrt((x2-x1)**2+(y2-y1)**2); cum_d+=d; dists.append(cum_d)
    if dists: plt.figure(figsize=(10, 5)); plt.plot(range(len(dists)), dists, label="Ball Dist (px)", color='g'); plt.xlabel("Frame"); plt.ylabel("Pixels"); plt.title("Ball Distance"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(graph_dir, "ball_dist.png")); plt.close(); print("- Ball distance graph saved.")
if not depth_df.empty and 'poss_d_smooth' in depth_df.columns and 'ball_d_smooth' in depth_df.columns:
    plt.figure(figsize=(12, 6))
    valid_ball = depth_df.dropna(subset=['ball_d_smooth']); valid_poss = depth_df.dropna(subset=['poss_d_smooth'])
    if not valid_ball.empty: plt.plot(valid_ball['frame'], valid_ball['ball_d_smooth'], label='Ball Depth', color='g', alpha=0.9)
    if not valid_poss.empty: plt.plot(valid_poss['frame'], valid_poss['poss_d_smooth'], label='Possessor Depth', color='r', alpha=0.9)
    else: print("- Warning: No valid possessor depth data to plot.")
    plt.xlabel("Frame"); plt.ylabel("Depth (MiDaS Scale)"); plt.title("Object Depth (Smoothed)"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "depth_plot_fixed.png")); plt.close(); print("- Depth graph saved.")
else: print("- Skipping depth graph (no valid data).")
summary_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
if ball_positions: pts = np.array([(int(p[0]), int(p[1])) for p in ball_positions]); cv2.polylines(summary_img, [pts], False, (0,255,255), 2)
for f_id, t_id, s_f, e_f in passes:
    s_pos = next((p for p in ball_positions if p[2] == s_f), None); e_pos = next((p for p in ball_positions if p[2] == e_f), None)
    if s_pos and e_pos: x1,y1,_=s_pos; x2,y2,_=e_pos; cv2.circle(summary_img,(int(x1),int(y1)),8,(0,255,0),-1); cv2.circle(summary_img,(int(x2),int(y2)),8,(0,0,255),-1)
cv2.imwrite(os.path.join(image_dir, "trajectory.png"), summary_img); print("- Trajectory image saved.")
print("\nProcessing finished.")