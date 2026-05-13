"""
Газрын зургийн асинхрон рендерер
"""
import threading
import time
import cv2
import numpy as np
from constants import MAP_W, MAP_H

try:
    from staticmap import StaticMap, CircleMarker, Line
    STATICMAP_OK = True
except ImportError:
    STATICMAP_OK = False


class MapRenderer:
    def __init__(self, w: int = MAP_W, h: int = MAP_H, zoom: int = 17):
        self.w    = w
        self.h    = h
        self.zoom = zoom
        self._img            = self._placeholder("MAP LOADING...")
        self._lock           = threading.Lock()
        self._busy           = False
        self._last_render_t  = 0.0
        self._render_interval = 3.0
        self.status = "READY"

    def request_render(self, cur_lat: float, cur_lon: float,
                       route=None, dest=None):
        now = time.time()
        if self._busy or (now - self._last_render_t) < self._render_interval:
            return
        self._last_render_t = now
        threading.Thread(
            target=self._worker,
            args=(cur_lat, cur_lon, route, dest),
            daemon=True,
        ).start()

    def get(self) -> np.ndarray:
        with self._lock:
            return self._img.copy()

    # ── Private ────────────────────────────────────────────────

    def _worker(self, cur_lat, cur_lon, route, dest):
        self._busy   = True
        self.status  = "RENDERING"
        try:
            if STATICMAP_OK:
                img = self._render(cur_lat, cur_lon, route, dest)
            else:
                img = self._placeholder("staticmap missing")
            with self._lock:
                self._img = img
            self.status = "READY"
        except Exception as e:
            with self._lock:
                self._img = self._placeholder(str(e)[:30])
            self.status = "ERROR"
        finally:
            self._busy = False

    def _render(self, cur_lat, cur_lon, route, dest) -> np.ndarray:
        m = StaticMap(
            self.w, self.h,
            url_template='https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'
        )
        if route and len(route) >= 2:
            m.add_line(Line([(p[1], p[0]) for p in route], "blue", 4))
        if dest and dest[0] is not None:
            m.add_marker(CircleMarker((dest[1], dest[0]), "red", 12))
        m.add_marker(CircleMarker((cur_lon, cur_lat), "white", 14))
        m.add_marker(CircleMarker((cur_lon, cur_lat), "#0066ff", 10))
        arr = cv2.cvtColor(np.array(m.render(zoom=self.zoom)), cv2.COLOR_RGB2BGR)
        cv2.rectangle(arr, (0, 0), (self.w - 1, self.h - 1), (80, 80, 80), 2)
        return arr

    def _placeholder(self, text: str) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 235, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (80, 80, 80), 2)
        for i in range(0, self.w, 40):
            cv2.line(img, (i, 0), (i, self.h), (210, 210, 210), 1)
        for i in range(0, self.h, 40):
            cv2.line(img, (0, i), (self.w, i), (210, 210, 210), 1)
        cv2.putText(img, text, (10, self.h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        return img
