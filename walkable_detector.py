"""
Walkable Path Detector — tactile хавтан байхгүй үед явж болох замыг хайна
"""
import cv2
import numpy as np
from constants import (
    WALK_ROI_TOP, WALK_GRID_W, WALK_GRID_H,
    WALK_TEX_THRESH, WALK_EDGE_THRESH, WALK_MIN_CELLS,
    WALK_SAT_MAX, WALK_VAL_MIN, MESSAGES, COLORS, ARROWS,
)


class WalkablePathDetector:
    """
    Аргачлал:
      1. ROI — frame-ийн доод хагас
      2. Grid cell тус бүрд: texture, edge density, HSV шалгана
      3. Walkable / not walkable шошголно
      4. 3 бүс (зүүн/дунд/баруун) зоны оноо гаргана
      5. Хамгийн чөлөөтэй бүсийг сонгоно
    """

    def __init__(self,
                 grid_w: int = WALK_GRID_W,
                 grid_h: int = WALK_GRID_H,
                 roi_top: float = WALK_ROI_TOP):
        self.grid_w  = grid_w
        self.grid_h  = grid_h
        self.roi_top = roi_top

    def analyze(self, frame: np.ndarray, tactile_mask: np.ndarray = None) -> dict:
        h, w = frame.shape[:2]
        y0   = int(h * self.roi_top)
        roi  = frame[y0:, :]
        rh, rw = roi.shape[:2]

        tac_roi = tactile_mask[y0:, :] if tactile_mask is not None else None

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 60, 160)
        lap   = cv2.Laplacian(gray, cv2.CV_32F)

        cell_w = max(rw // self.grid_w, 1)
        cell_h = max(rh // self.grid_h, 1)
        walkable = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                y1 = gy * cell_h
                y2 = (gy + 1) * cell_h if gy < self.grid_h - 1 else rh
                x1 = gx * cell_w
                x2 = (gx + 1) * cell_w if gx < self.grid_w - 1 else rw

                cell_lap   = lap[y1:y2, x1:x2]
                cell_edges = edges[y1:y2, x1:x2]
                cell_hsv   = hsv[y1:y2, x1:x2]

                tex_var      = float(np.var(cell_lap))
                edge_density = float(np.sum(cell_edges > 0)) / max(cell_edges.size, 1)
                sat_mean     = float(np.mean(cell_hsv[..., 1]))
                val_mean     = float(np.mean(cell_hsv[..., 2]))

                tac_overlap = 0.0
                if tac_roi is not None:
                    cell_tac    = tac_roi[y1:y2, x1:x2]
                    tac_overlap = float(np.sum(cell_tac > 0)) / max(cell_tac.size, 1)

                if tac_overlap > 0.15:
                    walkable[gy, gx] = 2  # tactile-аар баталгаажсан
                elif (tex_var < WALK_TEX_THRESH and
                      edge_density < WALK_EDGE_THRESH and
                      sat_mean < WALK_SAT_MAX and
                      val_mean > WALK_VAL_MIN):
                    walkable[gy, gx] = 1  # vision candidate

        third = self.grid_w // 3
        L = self._zone_score(walkable[:, :third])
        C = self._zone_score(walkable[:, third:2*third])
        R = self._zone_score(walkable[:, 2*third:])

        total_walk = int(np.sum(walkable > 0))
        confirmed  = int(np.sum(walkable == 2))

        direction, conf = self._decide(L, C, R, total_walk, confirmed)

        return {
            "direction":     direction,
            "conf":          conf,
            "message":       MESSAGES.get(direction, ""),
            "color":         COLORS.get(direction, (100, 100, 100)),
            "arrow":         ARROWS.get(direction, "?"),
            "zones":         (L, C, R),
            "total":         total_walk,
            "confirmed":     confirmed,
            "walkable_grid": walkable,
            "roi_top":       y0,
        }

    @staticmethod
    def _zone_score(zone: np.ndarray) -> float:
        if zone.size == 0:
            return 0.0
        weights = np.linspace(0.5, 1.5, zone.shape[0]).reshape(-1, 1)
        w_zone  = (zone > 0).astype(np.float32) * weights
        return float(np.sum(w_zone)) / float(np.sum(weights) * zone.shape[1])

    @staticmethod
    def _decide(L: float, C: float, R: float,
                total: int, confirmed: int):
        if total < WALK_MIN_CELLS:
            return "walk_unsafe", 0.0

        best   = max(L, C, R)
        margin = best - min(L, C, R)
        bonus  = confirmed / max(total, 1) * 0.3

        if L >= 0.50 and C >= 0.50 and R >= 0.50:
            return "walk_straight", min(C * 1.2 + bonus, 1.0)

        if best < 0.20 or margin < 0.10:
            return "walk_unsafe", 0.0

        if C >= 0.50 and C >= best - 0.15:
            direction = "walk_straight"
        elif best == L:
            direction = "walk_left"
        elif best == R:
            direction = "walk_right"
        else:
            direction = "walk_straight"

        return direction, min(best * 1.5 + bonus, 1.0)
