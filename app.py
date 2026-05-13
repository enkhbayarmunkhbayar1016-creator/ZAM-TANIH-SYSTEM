"""
Tactile Navigation — Flask Web Server
Утасны камераас frame авч, навигацийн заавар буцаана.
"""
import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

from constants import CONF, IMGSZ
from tactile_nav import TactileNavigator, get_mask
from path_manager import PathLostManager
from obstacle_detector import ObstacleYOLO
from walkable_detector import WalkablePathDetector
from bridge_manager import SafeBridgeManager
from decision_engine import DecisionEngine

app = Flask(__name__)

# ── Загваруудыг нэг удаа ачаалах ──────────────────────────────
# TACTILE_MODEL: тактил хавтас таних custom загвар (best.pt)
# OBSTACLE_MODEL: COCO саад таних загвар (yolov8n.pt)
TACTILE_MODEL  = os.environ.get("TACTILE_MODEL",  "best.pt")
OBSTACLE_MODEL = os.environ.get("OBSTACLE_MODEL", "yolov8n.pt")

print(f"[INIT] Тактил загвар: {TACTILE_MODEL}")
print(f"[INIT] Саадын загвар: {OBSTACLE_MODEL}")
try:
    from ultralytics import YOLO
    _tac_model = YOLO(TACTILE_MODEL)
    print("[INIT] ✅ Тактил загвар бэлэн")
except Exception as e:
    print(f"[INIT] ❌ Тактил загвар алдаа: {e}")
    _tac_model = None

_tac_nav   = TactileNavigator()
_path_mgr  = PathLostManager()
_obstacle  = ObstacleYOLO(OBSTACLE_MODEL)
_walk_det  = WalkablePathDetector()
_bridge    = SafeBridgeManager()
_decider   = DecisionEngine()

_walk_counter = 0
_last_walk    = None


def _decode_frame(data_url: str) -> np.ndarray | None:
    """Base64 data URL → OpenCV BGR frame."""
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _process(frame: np.ndarray) -> dict:
    global _walk_counter, _last_walk

    if _tac_model is None:
        return {"direction": "safety_stop", "message": "ZOGS | Model not loaded",
                "arrow": "[X]", "conf": 0.0, "mode": "ERROR",
                "color": [220, 0, 0]}

    h, w = frame.shape[:2]

    # Tactile inference
    res  = _tac_model(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    mask = get_mask(res, w, h)

    tactile_res           = _tac_nav.analyze(mask)
    tactile_res["pixels"] = int(np.sum(mask > 0))

    path_status, path_msg_key = _path_mgr.update(tactile_res["pixels"])

    # Obstacle (COCO)
    obstacle_res = _obstacle.detect(frame)

    # Walkable (хэдэн frame тутамд)
    _walk_counter += 1
    if _walk_counter % 3 == 0 or _last_walk is None:
        _last_walk = _walk_det.analyze(frame, tactile_mask=mask)

    # Шийдвэр
    final = _decider.decide(
        tactile_res, path_status, path_msg_key,
        obstacle_res, _last_walk, _bridge, None
    )

    return {
        "direction": final["direction"],
        "message":   final["message"],
        "arrow":     final["arrow"],
        "conf":      round(float(final["conf"]), 2),
        "mode":      final.get("mode", "?"),
        "color":     list(final["color"]),
        "pixels":    tactile_res["pixels"],
    }


# ── Routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/frame", methods=["POST"])
def api_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "image байхгүй"}), 400

    frame = _decode_frame(image_data)
    if frame is None:
        return jsonify({"error": "frame decode алдаа"}), 400

    result = _process(frame)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status":         "ok",
        "tactile_model":  TACTILE_MODEL,
        "obstacle_model": OBSTACLE_MODEL,
        "model_loaded":   _tac_model is not None,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
