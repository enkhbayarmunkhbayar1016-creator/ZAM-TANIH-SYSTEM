"""
Тактил навигацийн модуль
"""
import cv2
import numpy as np
from collections import Counter
from constants import (
    PIX_LOST, HIST_SIZE, DRIFT_T,
    MESSAGES, COLORS, ARROWS,
)


def get_mask(result, w: int, h: int) -> np.ndarray:
    """YOLO result-аас binary mask үүсгэнэ."""
    combined = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        return combined
    for m in result.masks.data:
        seg = m.cpu().numpy().astype(np.uint8)
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        combined = cv2.bitwise_or(combined, seg * 255)
    return combined


def shape_analysis(mask: np.ndarray):
    """Маскны хэлбэрийн дрифт утгыг тооцно."""
    h, w = mask.shape
    q = h // 4
    bands = [mask[i*q:(i+1)*q if i < 3 else h, :] for i in range(4)]

    def cx(band):
        ys, xs = np.where(band > 0)
        if len(xs) == 0:
            return None
        return (float(np.mean(xs)) - w / 2) / (w / 2)

    cxs   = [cx(b) for b in bands]
    valid = [(i, v) for i, v in enumerate(cxs) if v is not None]
    if len(valid) < 2:
        return 0.0, cxs
    top    = [v for i, v in valid if i <= 1]
    bottom = [v for i, v in valid if i >= 2]
    if not top or not bottom:
        return 0.0, cxs
    return float(np.mean(top)) - float(np.mean(bottom)), cxs


class TactileNavigator:
    def __init__(self):
        self.history    = []
        self.drift_hist = []

    def analyze(self, mask: np.ndarray) -> dict:
        h, w  = mask.shape
        total = int(np.sum(mask > 0))
        if total < PIX_LOST:
            return self._make("no_path", 0.0, 0.0, 0, 0, 0, 0.0, [], total)

        y0  = int(h * 0.45)
        roi = mask[y0:, :]
        rt  = max(int(np.sum(roi > 0)), 1)
        w3  = w // 3
        L = int(np.sum(roi[:, :w3]       > 0))
        C = int(np.sum(roi[:, w3:2*w3]   > 0))
        R = int(np.sum(roi[:, 2*w3:]     > 0))
        lr, cr, rr = L / rt, C / rt, R / rt

        drift, cxs = shape_analysis(mask)
        self.drift_hist.append(drift)
        if len(self.drift_hist) > 4:
            self.drift_hist.pop(0)
        smooth = float(np.mean(self.drift_hist))

        ys, xs = np.where(mask > 0)
        cx_all = (float(np.mean(xs)) - w / 2) / (w / 2) if len(xs) > 0 else 0.0

        if lr > 0.28 and rr > 0.28 and cr > 0.15:
            d, conf = "intersection", (lr + rr) / 2
        elif smooth < -DRIFT_T:
            d, conf = "turn_left",  min(abs(smooth) / 0.5, 1.0)
        elif smooth > DRIFT_T:
            d, conf = "turn_right", min(abs(smooth) / 0.5, 1.0)
        elif lr >= 0.18 and lr > rr and cr < 0.58:
            d, conf = "turn_left",  lr
        elif rr >= 0.18 and rr > lr and cr < 0.58:
            d, conf = "turn_right", rr
        elif cx_all < -0.30 and cr < 0.55:
            d, conf = "turn_left",  abs(cx_all)
        elif cx_all > 0.30 and cr < 0.55:
            d, conf = "turn_right", abs(cx_all)
        else:
            d, conf = "straight", max(cr, 0.5)

        if d == "straight":
            if cx_all < -0.32:
                d = "straight_left"
            elif cx_all > 0.32:
                d = "straight_right"

        self.history.append(d)
        if len(self.history) > HIST_SIZE:
            self.history.pop(0)
        stable = Counter(self.history).most_common(1)[0][0]

        return self._make(stable, min(conf, 1.0), cx_all,
                          lr, cr, rr, smooth, cxs, total)

    def _make(self, d, conf, off, l, c, r, drift, cxs, px) -> dict:
        return {
            "direction": d,
            "conf":      float(conf),
            "offset":    float(off),
            "message":   MESSAGES.get(d, ""),
            "color":     COLORS.get(d, (100, 100, 100)),
            "arrow":     ARROWS.get(d, "?"),
            "ratios":    (l, c, r),
            "drift":     float(drift),
            "cxs":       cxs,
            "pixels":    px,
        }
