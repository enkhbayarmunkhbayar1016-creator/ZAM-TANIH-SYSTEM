"""
GPS навигацийн модуль — OpenRouteService API
"""
import math
import traceback
import requests


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Хоёр цэгийн хоорондох метрийн зай."""
    R    = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RouteCalculator:
    GEOCODE_URL = "https://api.openrouteservice.org/geocode/search"
    ROUTE_URL   = "https://api.openrouteservice.org/v2/directions/foot-walking/geojson"

    def __init__(self, api_key: str):
        self.api_key          = api_key
        self.route            = None
        self.steps            = []
        self.dest_lat         = None
        self.dest_lon         = None
        self.dest_name        = None
        self.total_dist       = 0
        self.current_step_idx = 0
        self.status           = "NOT_READY"
        self.error_msg        = ""

    def has_key(self) -> bool:
        return bool(self.api_key) and len(self.api_key) > 20

    # ── Geocode ────────────────────────────────────────────────

    def geocode(self, place_name: str,
                focus_lat=None, focus_lon=None):
        if not self.has_key():
            self.error_msg = "API key байхгүй"
            return None
        params = {"api_key": self.api_key, "text": place_name, "size": 1}
        if focus_lat and focus_lon:
            params["focus.point.lat"] = focus_lat
            params["focus.point.lon"] = focus_lon
        try:
            print(f"[GEOCODE] '{place_name}' хайж байна...")
            r = requests.get(self.GEOCODE_URL, params=params, timeout=10)
            if r.status_code != 200:
                self.error_msg = f"Geocode HTTP {r.status_code}"
                return None
            feat = r.json().get("features", [])
            if not feat:
                self.error_msg = f"'{place_name}' олдсонгүй"
                return None
            coords = feat[0]["geometry"]["coordinates"]
            print(f"✅ Олдлоо: {feat[0]['properties'].get('label', place_name)}")
            return coords[1], coords[0]
        except Exception as e:
            self.error_msg = f"Geocode алдаа: {e}"
            return None

    # ── Route ──────────────────────────────────────────────────

    def calculate_route(self, start_lat: float, start_lon: float,
                        dest_lat=None, dest_lon=None,
                        dest_name: str = None) -> bool:
        if not self.has_key():
            self.status    = "ERROR"
            self.error_msg = "API key байхгүй"
            return False

        self.status = "CALCULATING"
        if dest_name and (dest_lat is None or dest_lon is None):
            geo = self.geocode(dest_name, start_lat, start_lon)
            if not geo:
                self.status = "ERROR"
                return False
            dest_lat, dest_lon = geo

        self.dest_lat  = dest_lat
        self.dest_lon  = dest_lon
        self.dest_name = dest_name or "GPS цэг"

        headers = {"Authorization": self.api_key,
                   "Content-Type": "application/json"}
        body = {
            "coordinates": [[start_lon, start_lat], [dest_lon, dest_lat]],
            "instructions": True,
            "language": "en",
        }
        try:
            r = requests.post(self.ROUTE_URL, json=body,
                              headers=headers, timeout=15)
            if r.status_code != 200:
                self.status    = "ERROR"
                self.error_msg = f"Route HTTP {r.status_code}"
                return False
            data  = r.json()
            feats = data.get("features", [])
            if not feats:
                self.status    = "ERROR"
                self.error_msg = "Зам олдсонгүй"
                return False

            geom       = feats[0]["geometry"]["coordinates"]
            self.route = [(c[1], c[0]) for c in geom]
            self.steps = [
                {
                    "distance":    st.get("distance", 0),
                    "instruction": st.get("instruction", ""),
                    "type":        st.get("type", -1),
                    "way_points":  st.get("way_points", []),
                }
                for seg in feats[0]["properties"].get("segments", [])
                for st  in seg.get("steps", [])
            ]
            self.current_step_idx = 0
            self.total_dist = feats[0]["properties"]["summary"]["distance"]
            self.status     = "OK"
            print(f"✅ Зам бэлэн: {self.total_dist:.0f}м, {len(self.steps)} алхам")
            return True
        except Exception as e:
            self.status    = "ERROR"
            self.error_msg = f"Route exception: {e}"
            traceback.print_exc()
            return False

    # ── Navigation ─────────────────────────────────────────────

    def get_next_instruction(self, cur_lat: float, cur_lon: float):
        if not self.route or self.status != "OK":
            return None
        dist_to_dest = haversine(cur_lat, cur_lon,
                                 self.dest_lat, self.dest_lon)
        if dist_to_dest < 15:
            return {"key": "gps_arrived", "dist": 0,
                    "remaining": 0, "arrived": True}

        nearest_dist = min(
            haversine(cur_lat, cur_lon, p[0], p[1]) for p in self.route
        )
        if nearest_dist > 30:
            return {"key": "gps_off_route",
                    "dist": int(nearest_dist),
                    "remaining": int(dist_to_dest),
                    "arrived": False}

        if self.current_step_idx >= len(self.steps):
            return {"key": "gps_straight", "dist": 0,
                    "remaining": int(dist_to_dest), "arrived": False}

        step = self.steps[self.current_step_idx]
        wp   = step.get("way_points", [])
        if wp and wp[0] < len(self.route):
            turn_lat, turn_lon = self.route[wp[0]]
            dist_to_turn = haversine(cur_lat, cur_lon, turn_lat, turn_lon)
            if dist_to_turn < 5:
                self.current_step_idx += 1
            stype = step.get("type", -1)
            if stype in (0, 6):    key = "gps_turn_left"
            elif stype in (1, 7):  key = "gps_turn_right"
            elif stype == 10:      key = "gps_arrived"
            else:                  key = "gps_straight"
            return {"key": key,
                    "dist": int(dist_to_turn),
                    "remaining": int(dist_to_dest),
                    "arrived": False}

        return {"key": "gps_straight", "dist": 0,
                "remaining": int(dist_to_dest), "arrived": False}
