"""
Саад илрүүлэлтийн модуль (YOLO COCO)
"""
from constants import COCO_DANGER, OBSTACLE_IMGSZ, OBSTACLE_CONF, OBSTACLE_INTERVAL


class ObstacleYOLO:
    def __init__(self, model_path: str = "yolov8n.pt"):
        print(f"[INIT] Саадын загвар: {model_path}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.ok = True
        except Exception as e:
            print(f"❌ Саадын загвар алдаа: {e}")
            self.model = None
            self.ok    = False
        self.last_result = None
        self.frame_count = 0

    def detect(self, frame, force: bool = False):
        if not self.ok:
            return None
        self.frame_count += 1
        if not force and self.frame_count % OBSTACLE_INTERVAL != 0:
            return self.last_result

        h, w = frame.shape[:2]
        try:
            results = self.model(
                frame, imgsz=OBSTACLE_IMGSZ,
                conf=OBSTACLE_CONF, verbose=False
            )[0]
        except Exception:
            return self.last_result

        dangers = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in COCO_DANGER:
                continue
            name, level = COCO_DANGER[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            area_ratio = (bw * bh) / (w * h)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if cy < h * 0.35:
                continue

            score = 0
            if area_ratio > 0.30:      score += 3
            elif area_ratio > 0.15:    score += 2
            elif area_ratio > 0.05:    score += 1
            else:                      continue

            if level == "high":        score += 2
            elif level == "medium":    score += 1

            if cx < w * 0.33:          position = "left"
            elif cx > w * 0.66:        position = "right"
            else:                      position = "center"

            dangers.append({
                "name":     name,
                "level":    level,
                "score":    score,
                "area":     area_ratio,
                "position": position,
                "bbox":     (int(x1), int(y1), int(x2), int(y2)),
                "conf":     float(box.conf[0]),
            })

        if not dangers:
            self.last_result = None
            return None

        top = max(dangers, key=lambda x: x["score"])
        if top["score"] >= 5:
            key = ("danger_high"   if top["position"] == "center" else
                   "danger_right"  if top["position"] == "left"   else "danger_left")
        elif top["score"] >= 3:
            key = ("danger_medium" if top["position"] == "center" else
                   "danger_right"  if top["position"] == "left"   else "danger_left")
        else:
            key = "danger_low"

        self.last_result = {"key": key, "top": top, "all": dangers}
        return self.last_result
