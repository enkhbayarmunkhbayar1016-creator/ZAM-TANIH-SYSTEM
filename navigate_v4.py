"""
═══════════════════════════════════════════════════════════════════
  ТАКТИЛ НАВИГАЦИ — БҮРЭН СИСТЕМ v4.0
  (Walkable Path + Safe Bridge Mode)
═══════════════════════════════════════════════════════════════════

АЖИЛЛУУЛАХ:
    python navigate_v4.py --model best.pt --dest "Сүхбаатарын талбай"

ТОВЧЛУУРУУД:
    q = гарах           m = маск
    d = debug           v = дуу ON/OFF
    r = reset           g = GPS зам
    w = walkable overlay
═══════════════════════════════════════════════════════════════════
"""
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, Future

import cv2
import numpy as np

# ── Dependency шалгах ─────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ pip install ultralytics")
    sys.exit(1)

# ── Дотоод модулиуд ───────────────────────────────────────────
from constants       import CONF, IMGSZ, CAM_W, CAM_H
from voice_module    import Voice
from tactile_nav     import TactileNavigator, get_mask
from path_manager    import PathLostManager
from obstacle_detector import ObstacleYOLO
from walkable_detector import WalkablePathDetector
from bridge_manager  import SafeBridgeManager
from gps_module      import RouteCalculator
from map_renderer    import MapRenderer
from decision_engine import DecisionEngine
from display         import draw_camera, draw_map_panel, combine_views

# ── API key ───────────────────────────────────────────────────
try:
    from config import ORS_API_KEY
    HAS_KEY = bool(ORS_API_KEY) and len(ORS_API_KEY) > 20
except ImportError:
    ORS_API_KEY = ""
    HAS_KEY = False

# ── Walkable-ийг хэдэн frame тутамд шинэчлэх ─────────────────
WALK_INTERVAL = 3


def main(tactile_model_path: str,
         dest: str          = None,
         dest_lat: float    = None,
         dest_lon: float    = None,
         cam_id: int        = 0,
         start_lat: float   = 47.9184,
         start_lon: float   = 106.9177):

    print("═" * 60)
    print("  ТАКТИЛ НАВИГАЦИ v4.0  (Walkable + Safe Bridge)")
    print("═" * 60)
    print(f"API key:   {'✅ Байгаа' if HAS_KEY else '❌ Байхгүй'}")
    print(f"Загвар:    {tactile_model_path}")
    print()

    # ── Загваруудыг эхлүүлэх ─────────────────────────────────
    tac_model = YOLO(tactile_model_path)
    tac_nav   = TactileNavigator()
    path_mgr  = PathLostManager()
    obstacle  = ObstacleYOLO("yolov8n.pt")
    walk_det  = WalkablePathDetector()
    bridge    = SafeBridgeManager()
    decider   = DecisionEngine()
    router    = RouteCalculator(ORS_API_KEY)
    map_rend  = MapRenderer()
    voice     = Voice()

    cur_lat, cur_lon = start_lat, start_lon

    if dest_lat is not None and dest_lon is not None:
        router.calculate_route(cur_lat, cur_lon,
                               dest_lat=dest_lat, dest_lon=dest_lon)
    elif dest and HAS_KEY:
        router.calculate_route(cur_lat, cur_lon, dest_name=dest)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("❌ Камер нээгдсэнгүй")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    show_mask  = True
    show_debug = True
    show_walk  = False
    fps        = 0.0
    prev_t     = time.time()
    prev_msg   = ""
    walk_frame = 0
    last_walk_res = None

    # ── ThreadPoolExecutor: tactile + obstacle зэрэгцээ ──────
    # max_workers=2: 2 YOLO загвар зэрэгцээ thread-д ажиллана
    # PyTorch inference GIL-ийг суллах тул бодит зэрэгцээ боломжтой
    executor = ThreadPoolExecutor(max_workers=2)

    print("\n✅ Систем эхэллээ!")
    print("   q=гарах  m=маск  d=debug  w=walkable  v=дуу  r=reset  g=GPS")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            # ── Зэрэгцээ inference: tactile + obstacle ─────────
            # frame-ийн хуулбар өгч race condition-оос зайлсхийнэ
            tac_frame = frame   # read-only хандалт — аюулгүй
            obs_frame = frame

            # Tactile inference + obstacle inference зэрэгцээ
            tac_future: Future = executor.submit(
                _run_tac_inference, tac_model, tac_frame, w, h
            )
            obs_future: Future = executor.submit(
                obstacle.detect, obs_frame
            )

            # ── Tactile mask авна (inference дууссан) ──────────
            mask         = tac_future.result()
            obstacle_res = obs_future.result()

            # State-тай TactileNavigator-г main thread-д ажиллуулна
            tactile_res           = tac_nav.analyze(mask)
            pixels                = int(np.sum(mask > 0))
            tactile_res["pixels"] = pixels

            path_status, path_msg_key = path_mgr.update(pixels)

            # ── Walkable (хэдэн frame тутамд) ──────────────────
            walk_frame += 1
            if walk_frame % WALK_INTERVAL == 0 or last_walk_res is None:
                last_walk_res = walk_det.analyze(frame, tactile_mask=mask)
            walkable_res = last_walk_res

            # ── GPS ────────────────────────────────────────────
            gps_res = router.get_next_instruction(cur_lat, cur_lon)

            # ── Шийдвэр ────────────────────────────────────────
            final = decider.decide(
                tactile_res, path_status, path_msg_key,
                obstacle_res, walkable_res, bridge, gps_res
            )

            # ── FPS тооцоо ─────────────────────────────────────
            now   = time.time()
            fps   = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
            prev_t = now
            final["fps"] = fps

            # ── Газрын зураг (асинхрон) ────────────────────────
            map_rend.request_render(
                cur_lat, cur_lon,
                router.route,
                (router.dest_lat, router.dest_lon) if router.dest_lat else None
            )

            # ── Дэлгэц ─────────────────────────────────────────
            cam_vis = draw_camera(
                frame, mask, final, fps,
                show_mask, show_debug,
                path_status, obstacle_res, voice.enabled,
                walkable_res, show_walk
            )
            map_vis = draw_map_panel(
                map_rend.get(), gps_res,
                router.status, router.dest_name, router.error_msg
            )
            combined = combine_views(cam_vis, map_vis)

            # ── Дуут мэдэгдэл ──────────────────────────────────
            if voice.enabled and final["message"] != prev_msg:
                voice.say(final["message"], final["direction"])
                prev_msg = final["message"]

            cv2.imshow("Tactile Navigation v4.0", combined)
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('m'): show_mask  = not show_mask
            elif key == ord('d'): show_debug = not show_debug
            elif key == ord('w'):
                show_walk = not show_walk
                print(f"Walkable overlay: {'ON' if show_walk else 'OFF'}")
            elif key == ord('v'):
                voice.enabled = not voice.enabled
                print(f"Дуу: {'ON' if voice.enabled else 'OFF'}")
            elif key == ord('r'):
                path_mgr.reset()
                bridge.cancel()
                print("PathManager + Bridge reset")
            elif key == ord('g'):
                if dest:
                    router.calculate_route(cur_lat, cur_lon, dest_name=dest)
                elif dest_lat is not None:
                    router.calculate_route(cur_lat, cur_lon,
                                           dest_lat=dest_lat, dest_lon=dest_lon)
    finally:
        executor.shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()
        print("Систем зогслоо.")


def _run_tac_inference(model, frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """YOLO inference + mask үүсгэлт — thread-д аюулгүй ажиллана.
    TactileNavigator (state-тай) main thread-д analyze хийнэ."""
    res  = model(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    return get_mask(res, w, h)


def main_cli():
    parser = argparse.ArgumentParser(
        description="Tactile Navigation v4.0"
    )
    parser.add_argument("--model",     default="best.pt",
                        help="Tactile YOLO загварын зам")
    parser.add_argument("--dest",      default=None,
                        help="Очих газрын нэр")
    parser.add_argument("--dest-lat",  type=float, default=None)
    parser.add_argument("--dest-lon",  type=float, default=None)
    parser.add_argument("--start-lat", type=float, default=47.9184)
    parser.add_argument("--start-lon", type=float, default=106.9177)
    parser.add_argument("--camera",    type=int,   default=0)
    args = parser.parse_args()
    main(
        args.model, args.dest,
        args.dest_lat, args.dest_lon,
        args.camera,
        args.start_lat, args.start_lon,
    )


if __name__ == "__main__":
    main_cli()
