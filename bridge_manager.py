"""
Safe Bridge Manager — tactile тасрах үед богино хугацааны аюулгүй үргэлжлэл
"""
import time
import numpy as np
from collections import Counter, deque
from constants import (
    BRIDGE_MAX_SEC, BRIDGE_MIN_FREE, BRIDGE_MIN_CONF, BRIDGE_NO_OBSTACLE,
)


class SafeBridgeManager:
    """
    Tactile тасрах үед:
      - сүүлийн найдвартай direction-ийг санана
      - obstacle + free-space шалгана
      - 1.5 секундээс илүү ажиллахгүй
    """

    def __init__(self,
                 max_sec:  float = BRIDGE_MAX_SEC,
                 min_free: float = BRIDGE_MIN_FREE,
                 min_conf: float = BRIDGE_MIN_CONF):
        self.max_sec  = max_sec
        self.min_free = min_free
        self.min_conf = min_conf

        self.active          = False
        self.bridge_start_t  = 0.0
        self.last_safe_t     = 0.0
        self.recent_dirs     = deque(maxlen=10)
        self.recent_confs    = deque(maxlen=10)

    # ── Public API ─────────────────────────────────────────────

    def update_tactile(self, tactile_res: dict):
        """Tactile сайн ажиллаж буй frame бүрт дуудна."""
        d    = tactile_res["direction"]
        conf = tactile_res["conf"]
        if d == "no_path":
            return
        self.recent_dirs.append(d)
        self.recent_confs.append(conf)
        self.last_safe_t = time.time()

    def try_bridge(self, walkable_res, obstacle_res):
        """
        Bridge mode-руу орох эсэхийг шийднэ.

        Буцаах:
          ("BRIDGE", "bridge_continue", conf, info)  — зөвшөөрөв
          ("STOP",   "bridge_stop",     0.0,  info)  — аюулгүй биш
        """
        now = time.time()

        if self.active and (now - self.bridge_start_t) > self.max_sec:
            self.active = False
            return ("STOP", "bridge_stop", 0.0,
                    {"reason": "bridge timeout"})

        best_dir, best_conf = self._best_saved_direction()
        if best_conf < self.min_conf:
            self.active = False
            return ("STOP", "bridge_stop", 0.0,
                    {"reason": f"low memory conf ({best_conf:.2f})"})

        if now - self.last_safe_t > 3.0:
            self.active = False
            return ("STOP", "bridge_stop", 0.0,
                    {"reason": "stale memory"})

        if BRIDGE_NO_OBSTACLE and obstacle_res:
            if obstacle_res["key"] in ("danger_high", "danger_medium"):
                self.active = False
                return ("STOP", "bridge_stop", 0.0,
                        {"reason": f"obstacle: {obstacle_res['key']}"})

        free_ratio = self._free_ratio(walkable_res, best_dir)
        if free_ratio is None:
            self.active = False
            return ("STOP", "bridge_stop", 0.0,
                    {"reason": "no free-space info"})
        if free_ratio < self.min_free:
            self.active = False
            return ("STOP", "bridge_stop", 0.0,
                    {"reason": f"low free space ({free_ratio:.2f})"})

        if not self.active:
            self.active         = True
            self.bridge_start_t = now

        elapsed   = now - self.bridge_start_t
        remaining = max(self.max_sec - elapsed, 0.0)
        return ("BRIDGE", "bridge_continue", best_conf,
                {"saved_dir": best_dir,
                 "elapsed":   elapsed,
                 "remaining": remaining,
                 "free":      free_ratio})

    def cancel(self):
        self.active = False

    # ── Private ────────────────────────────────────────────────

    def _best_saved_direction(self):
        if not self.recent_dirs:
            return "straight", 0.0
        norm = []
        for d, c in zip(self.recent_dirs, self.recent_confs):
            if d in ("straight", "straight_left", "straight_right"):
                norm.append(("straight", c))
            else:
                norm.append((d, c))
        counter  = Counter(d for d, _ in norm)
        best_dir = counter.most_common(1)[0][0]
        confs    = [c for d, c in norm if d == best_dir]
        return best_dir, float(np.mean(confs)) if confs else 0.0

    @staticmethod
    def _free_ratio(walkable_res, direction: str):
        if not walkable_res:
            return None
        L, C, R = walkable_res["zones"]
        if direction == "turn_left":
            return max(L, C)
        if direction == "turn_right":
            return max(R, C)
        return C
