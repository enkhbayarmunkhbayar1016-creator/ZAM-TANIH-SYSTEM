"""
Дэлгэцийн дүрслэлийн модуль
"""
import cv2
import numpy as np
from constants import PIX_NORMAL, PIX_WARNING, PIX_CRITICAL


# ── Walkable overlay ────────────────────────────────────────────

def draw_walkable_overlay(vis: np.ndarray, walkable_res: dict) -> np.ndarray:
    """Walkable cell-үүдийг ногоон/шар сүүдрээр харуулна."""
    if walkable_res is None:
        return vis
    grid = walkable_res.get("walkable_grid")
    if grid is None:
        return vis
    h, w = vis.shape[:2]
    y0   = walkable_res["roi_top"]
    gh, gw = grid.shape
    cell_h = max((h - y0) // gh, 1)
    cell_w = max(w // gw, 1)
    overlay = vis.copy()
    for gy in range(gh):
        for gx in range(gw):
            v = grid[gy, gx]
            if v == 0:
                continue
            y1 = y0 + gy * cell_h
            y2 = y0 + (gy + 1) * cell_h
            x1 = gx * cell_w
            x2 = (gx + 1) * cell_w
            col = (0, 220, 60) if v == 2 else (60, 200, 220)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
    return cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)


# ── Камерын дэлгэц ──────────────────────────────────────────────

def draw_camera(frame: np.ndarray, mask: np.ndarray, result: dict,
                fps: float, show_mask: bool, show_debug: bool,
                path_mode: str, obstacle_res, voice_on: bool,
                walkable_res, show_walk: bool) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    col  = result["color"]

    if show_mask and np.any(mask > 0):
        overlay = vis.copy()
        overlay[mask > 0] = [int(c * 0.65) for c in col]
        vis = cv2.addWeighted(vis, 0.45, overlay, 0.55, 0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 2)

    if show_walk and walkable_res is not None:
        vis = draw_walkable_overlay(vis, walkable_res)

    if obstacle_res:
        for d in obstacle_res["all"]:
            x1, y1, x2, y2 = d["bbox"]
            lvl = d["level"]
            box_col = ((0, 0, 255) if lvl == "high" else
                       (0, 140, 255) if lvl == "medium" else (0, 200, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_col, 2)
            cv2.putText(vis, f"{d['name']} {d['conf']:.0%}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, box_col, 1)

    if show_debug:
        _draw_debug(vis, h, w, result, path_mode, walkable_res, voice_on)

    vis = _draw_info_box(vis, h, w, col, result)
    vis = _draw_pixel_gauge(vis, w, result)
    return vis


def _draw_debug(vis, h, w, result, path_mode, walkable_res, voice_on):
    w3 = w // 3
    y0 = int(h * 0.45)
    lr, cr, rr = result["ratios"]
    cv2.line(vis, (w3,  y0), (w3,  h), (50, 50, 50), 1)
    cv2.line(vis, (2*w3, y0), (2*w3, h), (50, 50, 50), 1)
    cv2.line(vis, (0,   y0), (w,    y0), (50, 50, 50), 1)
    cv2.putText(vis, f"PATH:{path_mode}", (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 50), 1)
    cv2.putText(vis, f"L:{lr:.0%} C:{cr:.0%} R:{rr:.0%}",
                (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (120, 120, 120), 1)
    if walkable_res:
        wL, wC, wR = walkable_res["zones"]
        cv2.putText(vis, f"WALK L:{wL:.0%} C:{wC:.0%} R:{wR:.0%}",
                    (4, h - 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (180, 220, 80), 1)
    cv2.putText(vis, f"MODE:{result.get('mode', '?')}", (4, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 180, 60), 1)
    vt  = "VOICE:ON" if voice_on else "VOICE:OFF"
    vc  = (0, 220, 0) if voice_on else (100, 100, 100)
    cv2.putText(vis, vt, (w - 95, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, vc, 1)


def _draw_info_box(vis, h, w, col, result) -> np.ndarray:
    bh  = 95
    bg  = vis.copy()
    cv2.rectangle(bg, (0, 0), (w, bh), (8, 8, 8), -1)
    vis = cv2.addWeighted(vis, 0.2, bg, 0.8, 0)
    cv2.rectangle(vis, (0, 0), (w, bh), col, 2)
    cv2.putText(vis, result["arrow"], (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3, cv2.LINE_AA)
    parts  = result["message"].split(" | ")
    label  = parts[0]
    detail = parts[1] if len(parts) > 1 else ""
    cv2.putText(vis, label, (80, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    if detail:
        cv2.putText(vis, detail, (80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    cv2.putText(vis, f"conf:{result['conf']:.0%}  fps:{result.get('fps', 0):.0f}",
                (80, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (140, 140, 140), 1, cv2.LINE_AA)
    return vis


def _draw_pixel_gauge(vis, w, result) -> np.ndarray:
    pix = result.get("pixels", 0)
    bh  = 95
    gw  = int(min(pix / PIX_NORMAL, 1.0) * (w - 20))
    gc  = ((0, 220, 20) if pix > PIX_WARNING else
           (0, 200, 200) if pix > PIX_CRITICAL else (0, 30, 255))
    cv2.rectangle(vis, (10, bh + 3), (10 + gw, bh + 10), gc, -1)
    cv2.rectangle(vis, (10, bh + 3), (w - 10,  bh + 10), (50, 50, 50), 1)
    return vis


# ── Газрын зургийн дэлгэц ───────────────────────────────────────

def draw_map_panel(map_img: np.ndarray, gps_res,
                   route_status: str, dest_name: str,
                   error_msg: str) -> np.ndarray:
    if map_img is None:
        return None
    img  = map_img.copy()
    h, w = img.shape[:2]
    bar  = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 60), (15, 15, 15), -1)
    img  = cv2.addWeighted(img, 0.25, bar, 0.75, 0)

    status_map = {
        "NOT_READY":   ("GPS: BELEN BUS",      (150, 150, 150)),
        "CALCULATING": ("GPS: TOOTSOOLOJ...",   (200, 200, 0)),
        "OK":          ("GPS: AJILLAJ BAINA",   (0, 220, 100)),
        "ERROR":       ("GPS: ALDAA",           (0, 100, 255)),
    }
    txt, col = status_map.get(route_status, ("GPS: ?", (150, 150, 150)))
    cv2.putText(img, txt, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    if route_status == "ERROR" and error_msg:
        cv2.putText(img, error_msg[:35], (8, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
    elif dest_name:
        cv2.putText(img, f"-> {dest_name[:25]}", (8, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if gps_res and route_status == "OK":
        if gps_res["arrived"]:
            cv2.putText(img, "HURLEE!", (8, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
        else:
            cv2.putText(img, f"Uldsen: {gps_res['remaining']}m",
                        (8, 53), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 255), 1)
        if not gps_res["arrived"] and gps_res["dist"] > 0:
            bottom = img.copy()
            cv2.rectangle(bottom, (0, h - 40), (w, h), (15, 15, 15), -1)
            img = cv2.addWeighted(img, 0.25, bottom, 0.75, 0)
            arrow_map = {
                "gps_turn_left":  "<-- ZUUN",
                "gps_turn_right": "BARUUN -->",
                "gps_straight":   "^^ SHULUUN",
                "gps_off_route":  "ZAMAAS GARLAA!",
            }
            atxt = arrow_map.get(gps_res["key"], "")
            cv2.putText(img, f"{atxt} {gps_res['dist']}m",
                        (8, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 255, 100), 1)
    return img


def combine_views(cam_frame: np.ndarray,
                  map_frame: np.ndarray) -> np.ndarray:
    if map_frame is None:
        return cam_frame
    ch = cam_frame.shape[0]
    if map_frame.shape[0] != ch:
        map_frame = cv2.resize(
            map_frame, (map_frame.shape[1], ch)
        )
    return np.hstack([cam_frame, map_frame])
