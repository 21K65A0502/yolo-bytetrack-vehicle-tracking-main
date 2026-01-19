"""
Robust Object Tracking for a Specific Video with Vehicle Names
"""

import cv2
import numpy as np
import time

# Try to import ultralytics YOLO; fallback if unavailable
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    HAS_ULTRALYTICS = False

# Try ByteTrack tracker
try:
    from bytetrack.byte_track import ByteTrack
except Exception:
    try:
        from supervision.tracker.byte_track import ByteTrack  # type: ignore
    except Exception:
        ByteTrack = None

# YOLOv8 class names (COCO)
YOLO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Fallback motion detector
class FallbackMotionDetector:
    def __init__(self, min_area=500, history=500, varThreshold=16, detectShadows=True):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
        self.min_area = min_area

    def detect(self, frame):
        fg = self.bg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            dets.append({
                "bbox": [float(max(0, x)), float(max(0, y)), float(min(w-1, x+bw)), float(min(h-1, y+bh))],
                "score": float(min(1.0, area / (w * h))),
                "label": "motion",
            })
        return dets

# Main tracking class
class ObjectTracking:
    def __init__(self, model_path="yolov8s.pt", device="cpu"):
        self.device = device

        if HAS_ULTRALYTICS and YOLO is not None:
            try:
                print("Using ultralytics YOLO model.")
                self.model = YOLO(model_path)
                self.use_ultralytics = True
            except Exception:
                print("Failed to load YOLO, using fallback detector.")
                self.model = None
                self.use_ultralytics = False
                self.motion = FallbackMotionDetector()
        else:
            print("YOLO not available — using fallback motion detector.")
            self.model = None
            self.use_ultralytics = False
            self.motion = FallbackMotionDetector()

        # Tracker (optional)
        self.tracker = None
        if ByteTrack is not None:
            try:
                self.tracker = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
                print("ByteTrack tracker initialized.")
            except Exception:
                self.tracker = None

    def _detect_frame(self, frame):
        if self.use_ultralytics and self.model is not None:
            results = self.model(frame, verbose=False)
            res = results[0] if isinstance(results, (list, tuple)) else results
            dets = []
            try:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                for i, box in enumerate(boxes):
                    class_idx = int(classes[i])
                    dets.append({
                        "bbox": box.tolist(),
                        "score": float(scores[i]),
                        "label": YOLO_CLASSES[class_idx] if class_idx < len(YOLO_CLASSES) else str(class_idx),
                    })
            except Exception:
                return []
            return dets
        else:
            return self.motion.detect(frame)

    def _annotate(self, frame, detections):
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            label = d.get("label", "")
            score = d.get("score", 0.0)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{label} {score:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def process(self, source, show=True, save_path=None):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detections = self._detect_frame(frame)

            tracked = []
            if self.tracker and len(detections) > 0:
                try:
                    bboxes = np.array([d["bbox"] for d in detections], dtype=float)
                    scores = np.array([d["score"] for d in detections], dtype=float)
                    classes = np.array([d.get("label", "") for d in detections])
                    tracks = self.tracker.update(bboxes, scores, classes)
                    for t in tracks:
                        if len(t) >= 6:
                            x1, y1, x2, y2, tid, cls = t[:6]
                            tracked.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "id": int(tid),
                                "label": str(cls),
                                "score": 0.0,
                            })
                except Exception:
                    tracked = []

            to_annot = tracked if tracked else detections
            out_frame = self._annotate(frame, to_annot)

            if show:
                cv2.imshow("Tracking", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer:
                writer.write(out_frame)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        total = time.time() - start_time
        print(f"Processed {frame_idx} frames in {total:.2f}s ({frame_idx/total:.2f} FPS)")

# ----------------------
# Run script
# ----------------------
if __name__ == "__main__":
    video_path = r"C:\Users\govar\OneDrive\Desktop\New folder\project\ByteTrack\yolo-bytetrack-vehicle-tracking-main\assets\video\vehicle_counting.mp4"
    save_output_path = r"C:\Users\govar\OneDrive\Desktop\New folder\project\ByteTrack\yolo-bytetrack-vehicle-tracking-main\out.mp4"

    ot = ObjectTracking(model_path="yolov8s.pt", device="cpu")
    ot.process(source=video_path, show=True, save_path=save_output_path)
