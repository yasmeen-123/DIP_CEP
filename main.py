import cv2
import numpy as np
import time
import threading
import os
import argparse
import importlib
from collections import deque
from metrics import PerformanceEvaluator

# ===============================
# YOLO
# ===============================
try:
    YOLO = getattr(importlib.import_module("ultralytics"), "YOLO")
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False


# ===============================
# LANE DETECTOR
# ===============================
class LaneDetector:
    def __init__(self):
        self.left_hist = deque(maxlen=16)
        self.right_hist = deque(maxlen=16)
        self.last_lane_valid = False
        self.road_segment_history = deque(maxlen=8)  # Store road detections across frames
        self.persistent_roads = {}  # Track stable roads
        self.debug_log = os.path.join(os.getcwd(), "debug_frames", "road_debug.log")

    def get_roi(self, img):
        h, w = img.shape[:2]

        pts = np.array([[
            (int(w * 0.18), h),
            (int(w * 0.82), h),
            (int(w * 0.62), int(h * 0.60)),
            (int(w * 0.38), int(h * 0.60))
        ]], dtype=np.int32)

        mask = np.zeros_like(img)
        cv2.fillPoly(mask, pts, 255)

        return cv2.bitwise_and(img, mask)

    def preprocess(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 40, 80])
        upper_yellow = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # White lane paint in bright regions.
        lower_white = np.array([0, 0, 185])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use slightly wider Canny range for this video
        edges = cv2.Canny(blur, 30, 140)

        lane_mask = cv2.bitwise_or(yellow_mask, white_mask)
        combined = cv2.bitwise_or(edges, lane_mask)
        # Reduce speckle and bridge small gaps
        kernel = np.ones((5, 5), dtype=np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        roi = self.get_roi(combined)

        return roi, edges

    def detect_lines(self, roi):
        return cv2.HoughLinesP(
            roi,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=40,
            maxLineGap=120
        )

    def classify(self, lines, width, height):
        left, right = [], []

        if lines is None:
            return left, right

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # allow very shallow slopes (ignore near-horizontal noise)
            if abs(slope) < 0.05:
                continue

            midx = (x1 + x2) / 2
            midy = (y1 + y2) / 2

            # prefer lower-half lines for lane cues
            if midy < (0.35 * height):
                # too high in image, skip
                continue

            if slope < 0 and midx < width / 2:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and midx > width / 2:
                right.append((x1, y1, x2, y2))

        return left, right

    def average(self, lines, h):
        if not lines:
            return None

        xs, ys = [], []

        for x1, y1, x2, y2 in lines:
            xs += [x1, x2]
            ys += [y1, y2]

        try:
            fit = np.polyfit(ys, xs, 1)
        except Exception:
            return None

        # evaluate within observed y-range to avoid huge extrapolation
        y_min = int(min(ys))
        y_max = int(max(ys))
        y1 = int(np.clip(h, y_min, y_max))
        y2 = int(np.clip(int(h * 0.60), y_min, y_max))

        x1 = int(np.polyval(fit, y1))
        x2 = int(np.polyval(fit, y2))

        # if extrapolation produces unrealistic span, treat as invalid
        if abs(x2 - x1) > int(0.9 * 960):
            return None

        # clamp to image bounds to avoid small overshoot
        x1 = int(np.clip(x1, 0, 960))
        x2 = int(np.clip(x2, 0, 960))

        return (x1, y1, x2, y2)

    def smooth(self, line, history):
        if line is not None:
            history.append(line)

        if not history:
            return None

        arr = np.array(history, dtype=np.float32)
        smoothed = tuple(np.mean(arr, axis=0).astype(int).tolist())
        return smoothed

    def validate_lane(self, left, right, width):
        if not left or not right:
            return False

        # bottom measurement (near vehicle)
        lane_width_bottom = abs(right[0] - left[0])
        lane_width_top = abs(right[2] - left[2])

        # sanity: left should be left of right at bottom
        if left[0] >= right[0]:
            return False

        if lane_width_bottom < int(width * 0.12) or lane_width_bottom > int(width * 0.85):
            return False

        if lane_width_top < int(width * 0.06) or lane_width_top > int(width * 0.75):
            return False

        return True

    def detect_road_segments(self, lines, left, right, width, height, overlay=None):
        """Detect emerging roads (excluding main lane)."""
        if lines is None or len(lines) == 0:
            return []

        segments = []
        lane_margin = int(width * 0.08)

        def lane_x_at(line, y):
            x1, y1, x2, y2 = line
            if y2 == y1:
                return None
            t = (y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)

        # Collect candidate road lines (not part of main lane)
        candidate_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            # skip near-horizontal noise (stricter threshold to reject noise)
            if abs(slope) < 0.15:
                continue

            midx = (x1 + x2) / 2
            midy = (y1 + y2) / 2
            # allow lines a bit higher to capture emerging side roads
            if midy < (0.25 * height):
                continue
            
            # Filter out main lane lines
            in_main_lane = False
            
            if left and right:
                left_x = lane_x_at(left, midy)
                right_x = lane_x_at(right, midy)

                if left_x is not None and right_x is not None:
                    lane_left = min(left_x, right_x) - lane_margin
                    lane_right = max(left_x, right_x) + lane_margin
                    if lane_left <= midx <= lane_right:
                        in_main_lane = True

            if not in_main_lane and left:
                main_left_slope = (left[3] - left[1]) / (left[2] - left[0] + 1e-6) if (left[2] - left[0]) != 0 else 0
                if abs(slope - main_left_slope) < 0.20:
                    in_main_lane = True

            if not in_main_lane and right:
                main_right_slope = (right[3] - right[1]) / (right[2] - right[0] + 1e-6) if (right[2] - right[0]) != 0 else 0
                if abs(slope - main_right_slope) < 0.20:
                    in_main_lane = True
            
            if not in_main_lane:
                candidate_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'slope': slope,
                    'midx': midx,
                    'length': np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                })
        
        if not candidate_lines:
            # log empty for frame
            try:
                os.makedirs(os.path.dirname(self.debug_log), exist_ok=True)
                with open(self.debug_log, "a") as f:
                    f.write(f"frame {getattr(self, 'last_frame_idx', -1)}: candidates=0 groups=0 segments=0\n")
            except Exception:
                pass
            return []
        
        # Cluster lines by slope and position - use very loose grouping to allow single lines
        processed_groups = {}
        for line_data in candidate_lines:
            slope_key = round(line_data['slope'] * 2) / 2  # very loose grouping
            pos_key = round(line_data['midx'] / 100) * 100
            group_key = (slope_key, pos_key)
            
            if group_key not in processed_groups:
                processed_groups[group_key] = []
            processed_groups[group_key].append(line_data)
        
        # Create segments from groups (accept even single strong lines)
        for group_key, group_lines in processed_groups.items():
            xs, ys = [], []
            total_length = 0
            
            for line_data in group_lines:
                x1, y1, x2, y2 = line_data['coords']
                xs += [x1, x2]
                ys += [y1, y2]
                total_length += line_data['length']
            
            # Skip if total line length is too small (very permissive)
            if total_length < 5:
                continue
            try:
                fit = np.polyfit(ys, xs, 1)

                # evaluate line at fixed positions to get consistent segments
                y1_seg = int(height)
                y2_seg = int(height * 0.55)

                x1_seg = int(np.polyval(fit, y1_seg))
                x2_seg = int(np.polyval(fit, y2_seg))

                # Clamp to valid range (no width filtering - accept all segments)
                x1_seg = int(np.clip(x1_seg, 0, width))
                x2_seg = int(np.clip(x2_seg, 0, width))

                segments.append({
                    'line': (x1_seg, y1_seg, x2_seg, y2_seg),
                    'direction': 'LEFT' if fit[0] < 0 else 'RIGHT',
                    'slope': fit[0],
                    'count': len(group_lines),
                    'strength': total_length
                })
            except:
                pass
        # draw candidate lines and write debug log
        try:
            os.makedirs(os.path.dirname(self.debug_log), exist_ok=True)
            with open(self.debug_log, "a") as f:
                f.write(f"frame {getattr(self, 'last_frame_idx', -1)}: candidates={len(candidate_lines)} groups={len(processed_groups)} segments={len(segments)}\n")
                # log top candidate summaries
                for i, c in enumerate(candidate_lines[:8]):
                    f.write(f"  cand{i}: midx={int(c['midx'])} slope={c['slope']:.3f} len={int(c['length'])}\n")
                for i, s in enumerate(segments):
                    f.write(f"  seg{i}: line={s['line']} dir={s['direction']} slope={s['slope']:.3f} cnt={s['count']} str={int(s['strength'])}\n")
        except Exception:
            pass

        if overlay is not None:
            for c in candidate_lines:
                x1, y1, x2, y2 = map(int, c['coords'])
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # Sort by strength and return top 2
        segments.sort(key=lambda item: item['strength'], reverse=True)
        return segments[:2]

    def smooth_road_segments(self, current_segments):
        """
        Apply light temporal smoothing to road segments to reduce flickering.
        """
        # Add current frame's segments to history
        self.road_segment_history.append(current_segments)
        
        # If not enough history, return current segments as-is
        if len(self.road_segment_history) < 2:
            return current_segments[:2]
        
        # Use only recent 5 frames for smoothing
        recent_history = list(self.road_segment_history)[-5:]
        
        # Count how often each road appears (by direction and approximate slope)
        road_votes = {}
        
        for frame_segments in recent_history:
            for seg in frame_segments:
                slope_key = round(seg['slope'] * 10) / 10  # Looser slope grouping
                direction = seg['direction']
                key = (direction, slope_key)
                
                if key not in road_votes:
                    road_votes[key] = {'count': 0, 'segments': []}
                
                road_votes[key]['count'] += 1
                road_votes[key]['segments'].append(seg)
        
        # Return roads that appear in any frame (show all detected segments)
        stable_segments = []
        
        for key, data in road_votes.items():
            if data['count'] >= 2:
                # Average the recent occurrences
                recent_segs = data['segments'][-3:]
                avg_line = self._average_road_lines([seg['line'] for seg in recent_segs])
                avg_slope = np.mean([seg['slope'] for seg in recent_segs])
                
                stable_segments.append({
                    'line': avg_line,
                    'direction': key[0],
                    'slope': avg_slope,
                    'count': data['count']
                })
            else:
                # Show even single-frame detections (no persistence requirement)
                seg = data['segments'][0]
                stable_segments.append(seg)
        
        stable_segments.sort(key=lambda item: item['count'], reverse=True)
        return stable_segments[:2]
    
    def _average_road_lines(self, lines):
        """Average multiple line coordinates."""
        if not lines:
            return None
        
        lines_arr = np.array(lines, dtype=np.float32)
        avg_line = tuple(np.mean(lines_arr, axis=0).astype(int).tolist())
        return avg_line

    def detect(self, frame, frame_idx=0):
        h, w = frame.shape[:2]
        self.last_frame_idx = frame_idx

        roi, edges = self.preprocess(frame)

        lines = self.detect_lines(roi)

        left_raw, right_raw = self.classify(lines, w, h)

        left = self.average(left_raw, h)
        right = self.average(right_raw, h)
        left = self.smooth(left, self.left_hist)
        right = self.smooth(right, self.right_hist)
        lane_valid = self.validate_lane(left, right, w)
        self.last_lane_valid = lane_valid

        overlay = frame.copy()

        # detect all road segments including emerging roads (draw candidates on overlay)
        road_segments = self.detect_road_segments(lines, left, right, w, h, overlay=overlay)
        
        # Apply temporal smoothing to road segments
        road_segments = self.smooth_road_segments(road_segments)

        # Main lane always counts as road 1 if valid
        # Side roads detected by detect_road_segments are additional
        side_roads = road_segments if road_segments else []
        main_road_valid = lane_valid and left and right
        
        # Total roads = main (if valid) + side roads
        total_roads = (1 if main_road_valid else 0) + len(side_roads)

        if left:
            cv2.line(overlay, left[:2], left[2:], (0, 255, 0), 5)

        if right:
            cv2.line(overlay, right[:2], right[2:], (0, 255, 0), 5)

        if left and right:
            poly = np.array([
                left[:2],
                left[2:],
                right[2:],
                right[:2]
            ])

            fill_color = (0, 190, 80) if lane_valid else (0, 120, 180)
            cv2.fillPoly(overlay, [poly], fill_color)

        # draw at most the strongest side roads so the HUD stays readable
        for seg in road_segments[:2]:
            x1, y1, x2, y2 = seg['line']
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 3)
            # small arrow for direction
            dx, dy = x2 - x1, y2 - y1
            if dx != 0 or dy != 0:
                norm = np.sqrt(dx**2 + dy**2) + 1e-6
                arrow_scale = 20
                arrow_x = int(x1 + dx * 0.5 + (-dy / norm) * arrow_scale)
                arrow_y = int(y1 + dy * 0.5 + (dx / norm) * arrow_scale)
                cv2.arrowedLine(overlay, (x1, y1), (arrow_x, arrow_y), (255, 255, 0), 2, tipLength=0.3)

        final = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        return final, left, right, edges, roi, lane_valid, total_roads, side_roads
# ===============================
# OBJECT DETECTOR
# ===============================
class ObstacleDetector:
    def __init__(self, use_yolo=False):
        self.model = None
        self.tracks = {}
        self.next_track_id = 1
        self.max_missing = 4
        self.min_hits = 1  # Show objects immediately on first detection
        self.iou_match_threshold = 0.30
        self.confidence_threshold = 0.20  # lower confidence to catch more objects
        self.skip_frames = 2  # run YOLO every N frames (reduce CPU/GPU load)
        self.last_detections = []
        self.last_detect_frame = -999
        self.det_downscale = 0.75  # perform detection on a slightly smaller image (higher res)
        self.min_box_area = 300  # smaller area threshold to include smaller objects
        self.allowed_classes = None  # None = allow all classes; set list to restrict
        # Worker thread fields
        self._worker_thread = None
        self._worker_lock = threading.Lock()
        self._worker_stop = threading.Event()
        self._worker_frame = None
        self._worker_frame_idx = -1
        self._worker_pending = False

        if use_yolo and YOLO_AVAILABLE:
            self.model = YOLO("yolov8n.pt")
            try:
                # prefer GPU when available
                self.model.to("cuda")
            except Exception:
                pass
            # start background worker to run detection
            self._worker_thread = threading.Thread(target=self._detection_worker, daemon=True)
            self._worker_thread.start()

        print("YOLO Loaded:", self.model is not None)

    def iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area

        if union == 0:
            return 0.0

        return inter_area / union

    def _match_tracks(self, detections):
        unmatched_detections = set(range(len(detections)))
        matched_tracks = set()

        candidates = []
        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                score = self.iou(track["bbox"], det)
                if score >= self.iou_match_threshold:
                    candidates.append((score, track_id, det_idx))

        for _, track_id, det_idx in sorted(candidates, reverse=True):
            if track_id in matched_tracks or det_idx not in unmatched_detections:
                continue

            self.tracks[track_id]["bbox"] = detections[det_idx]
            self.tracks[track_id]["hits"] += 1
            self.tracks[track_id]["missing"] = 0
            matched_tracks.add(track_id)
            unmatched_detections.remove(det_idx)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track["missing"] += 1
                if track["missing"] > self.max_missing:
                    del self.tracks[track_id]

        for det_idx in unmatched_detections:
            self.tracks[self.next_track_id] = {
                "bbox": detections[det_idx],
                "hits": 1,
                "missing": 0
            }
            self.next_track_id += 1

    def _detection_worker(self):
        """Background worker that runs YOLO on the latest submitted frame."""
        while not self._worker_stop.is_set():
            frame = None
            frame_idx = -1
            with self._worker_lock:
                if self._worker_pending and self._worker_frame is not None:
                    frame = self._worker_frame.copy()
                    frame_idx = self._worker_frame_idx
                    # mark as taken
                    self._worker_pending = False

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                h, w = frame.shape[:2]
                ds = self.det_downscale
                small = cv2.resize(frame, (int(w * ds), int(h * ds)))

                results = self.model(small, verbose=False, conf=self.confidence_threshold)
                new_dets = []

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if self.allowed_classes is not None and cls not in self.allowed_classes:
                            continue
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        x1 = int(x1 / ds)
                        y1 = int(y1 / ds)
                        x2 = int(x2 / ds)
                        y2 = int(y2 / ds)
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area < self.min_box_area:
                            continue
                        new_dets.append((x1, y1, x2, y2))

                # update cache
                with self._worker_lock:
                    self.last_detections = new_dets
                    self.last_detect_frame = frame_idx
            except Exception:
                # keep previous detections on error
                pass

        # worker exiting
        return

    def submit_frame(self, frame, frame_idx=0):
        """Submit current frame for background detection (non-blocking)."""
        if not self.model or self._worker_thread is None:
            return

        with self._worker_lock:
            self._worker_frame = frame.copy()
            self._worker_frame_idx = frame_idx
            self._worker_pending = True

    def stop(self):
        if self._worker_thread is not None:
            self._worker_stop.set()
            self._worker_thread.join(timeout=1.0)

    def _stable_obstacles(self):
        stable = []
        for track in self.tracks.values():
            if track["hits"] >= self.min_hits and track["missing"] <= 1:
                stable.append(track["bbox"])
        return stable

    def detect(self, frame, frame_idx=0):
        h, w = frame.shape[:2]
        detections = list(self.last_detections)

        self._match_tracks(detections)
        obstacles = self._stable_obstacles()

        danger_zone_top = int(h * 0.52)
        danger_zone_left = int(w * 0.30)
        danger_zone_right = int(w * 0.70)
        danger_count = 0

        for x1, y1, x2, y2 in obstacles:
            cx = (x1 + x2) // 2
            if y2 >= danger_zone_top and danger_zone_left <= cx <= danger_zone_right:
                danger_count += 1
                color = (0, 0, 255)
            else:
                color = (0, 160, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

        cv2.rectangle(
            frame,
            (danger_zone_left, danger_zone_top),
            (danger_zone_right, h - 5),
            (80, 80, 255),
            1
        )

        return frame, obstacles, danger_count


# ===============================
# DECISION
# ===============================
class DecisionEngine:
    def decide(self, left, right, danger_count, width, lane_valid):

        if danger_count > 0:
            return "STOP"

        if lane_valid and left and right:
            center = (left[0] + right[0]) / 2
            offset = center - width / 2

            if offset < -80:
                return "LEFT"
            elif offset > 80:
                return "RIGHT"
            else:
                return "FORWARD"

        return "HOLD"


# ===============================
# HUD
# ===============================
class HUD:
    def draw(self, frame, decision, fps, left, right, obs, lane_valid, total_road_count=0):
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)

        lane_status = f"L:{'YES' if left else 'NO'} R:{'YES' if right else 'NO'}"
        road_count = total_road_count

        cv2.putText(
            frame,
            f"Decision: {decision}",
            (20, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            lane_status,
            (300, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Objects: {len(obs)}",
            (550, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Roads: {road_count}",
            (750, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        if lane_valid:
            center_x = w // 2
            base_y = h - 120
            tip_y = h - 180

            if decision == "LEFT":
                cv2.arrowedLine(frame, (center_x, base_y), (center_x - 90, tip_y), (0, 255, 255), 4, tipLength=0.35)
            elif decision == "RIGHT":
                cv2.arrowedLine(frame, (center_x, base_y), (center_x + 90, tip_y), (0, 255, 255), 4, tipLength=0.35)
            elif decision == "FORWARD":
                cv2.arrowedLine(frame, (center_x, base_y), (center_x, tip_y), (0, 255, 255), 4, tipLength=0.35)

        return frame


# ===============================
# MAIN
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="college_video.mp4")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable YOLOv8 obstacle detection"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    lane = LaneDetector()
    obstacle = ObstacleDetector(use_yolo=args.yolo)
    decision_engine = DecisionEngine()
    hud = HUD()
    metrics = PerformanceEvaluator()

    fps_hist = deque(maxlen=20)
    frame_idx = 0
    debug_dir = os.path.join(os.getcwd(), "debug_frames")
    os.makedirs(debug_dir, exist_ok=True)

    while True:
        start = time.time()

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        lane_frame, left, right, edges, roi, lane_valid, total_roads, side_roads = lane.detect(frame, frame_idx=frame_idx)

        # submit frame for async detection and get immediate results from cache
        obstacle.submit_frame(lane_frame, frame_idx=frame_idx)
        obj_frame, obstacles, danger_count = obstacle.detect(lane_frame)

        decision = decision_engine.decide(
            left,
            right,
            danger_count,
            frame.shape[1],
            lane_valid
        )

        fps = 1 / (time.time() - start)
        fps_hist.append(fps)

        avg_fps = np.mean(fps_hist)

        metrics.log_frame(
            avg_fps,
            lane_valid and left is not None,
            lane_valid and right is not None,
            len(obstacles),
            decision
        )

        final = hud.draw(
            obj_frame,
            decision,
            avg_fps,
            left,
            right,
            obstacles,
            lane_valid,
            total_road_count=total_roads
        )

        # Save debug frame every 50 frames when lane not valid
        if not lane_valid and (frame_idx % 50) == 0:
            debug_path = os.path.join(debug_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(debug_path, final)
        frame_idx += 1

        cv2.imshow("Self Driving", final)
        cv2.imshow("Edges", edges)
        cv2.imshow("ROI", roi)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    metrics.save_report()
    cap.release()
    # stop background worker if any
    try:
        obstacle.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()