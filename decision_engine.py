"""
Decision Engine — Mode Machine: TACTILE / BRIDGE / WALKABLE / STOP
"""
from constants import (
    PIX_CRITICAL, SAFETY_MIN_CONF, STOP_ON_UNKNOWN,
    WALK_MIN_CONF, MESSAGES, COLORS, ARROWS,
)


class DecisionEngine:
    """
    Шийдвэрийн иерархи:
      1. DANGER_HIGH       → ZOGS  (бүх юмыг override)
      2. Tactile сайн      → TACTILE mode
      3. Tactile алдагдсан → BRIDGE → WALKABLE → STOP
      4. Тодорхойгүй       → STOP
    """

    def __init__(self):
        self.last_mode = "INIT"

    def decide(self, tactile_res: dict, path_status: str,
               path_msg_key: str, obstacle_res, walkable_res,
               bridge_mgr, gps_res) -> dict:

        # ── 1. Danger HIGH ──────────────────────────────────────
        if obstacle_res and obstacle_res["key"] == "danger_high":
            self.last_mode = "STOP_DANGER"
            key = "danger_high"
            msg = MESSAGES[key] + f" ({obstacle_res['top']['name']})"
            return self._pack(key, msg, tactile_res,
                              mode="STOP_DANGER", conf=1.0)

        # ── 2. Tactile сайн ─────────────────────────────────────
        tactile_ok = (
            tactile_res["direction"] != "no_path" and
            tactile_res["pixels"] > PIX_CRITICAL and
            tactile_res["conf"] >= SAFETY_MIN_CONF
        )

        if tactile_ok and path_status in ("NORMAL", "WARNING", "ENDING", "FOUND"):
            bridge_mgr.update_tactile(tactile_res)

            if obstacle_res and obstacle_res["key"] == "danger_medium":
                self.last_mode = "STOP_DANGER"
                key = "danger_medium"
                msg = MESSAGES[key] + f" ({obstacle_res['top']['name']})"
                return self._pack(key, msg, tactile_res,
                                  mode="STOP_DANGER", conf=1.0)

            if path_status == "WARNING":
                self.last_mode = "TACTILE"
                return self._pack("path_warning", MESSAGES["path_warning"],
                                  tactile_res, mode="TACTILE",
                                  conf=tactile_res["conf"])

            if path_status == "ENDING":
                self.last_mode = "TACTILE"
                return self._pack("path_ending", MESSAGES["path_ending"],
                                  tactile_res, mode="TACTILE",
                                  conf=tactile_res["conf"])

            if obstacle_res and obstacle_res["key"] in (
                    "danger_left", "danger_right", "danger_low"):
                key = obstacle_res["key"]
                msg = MESSAGES[key] + f" ({obstacle_res['top']['name']})"
                self.last_mode = "TACTILE"
                return self._pack(key, msg, tactile_res,
                                  mode="TACTILE", conf=tactile_res["conf"])

            self.last_mode = "TACTILE"
            out = tactile_res.copy()
            out["mode"] = "TACTILE"
            self._apply_gps_overlay(out, gps_res)
            return out

        # ── 3. Tactile алдагдсан ────────────────────────────────
        if path_status in ("LOST", "SEARCH", "FAILED"):
            bridge_result, bridge_key, bridge_conf, bridge_info = \
                bridge_mgr.try_bridge(walkable_res, obstacle_res)

            if bridge_result == "BRIDGE":
                self.last_mode = "BRIDGE"
                saved_dir  = bridge_info["saved_dir"]
                remaining  = bridge_info["remaining"]
                msg = f"{MESSAGES['bridge_continue']} ({saved_dir}, {remaining:.1f}s)"
                pack = self._pack("bridge_continue", msg, tactile_res,
                                  mode="BRIDGE", conf=bridge_conf)
                pack["arrow"]       = ARROWS.get(saved_dir, "...")
                pack["bridge_info"] = bridge_info
                return pack

            if (walkable_res and
                    walkable_res["direction"] != "walk_unsafe" and
                    walkable_res["conf"] >= WALK_MIN_CONF and
                    not (obstacle_res and obstacle_res["key"] in
                         ("danger_high", "danger_medium"))):
                self.last_mode = "WALKABLE"
                wd  = walkable_res["direction"]
                wcf = walkable_res["conf"]
                return self._pack(wd, MESSAGES[wd], tactile_res,
                                  mode="WALKABLE", conf=wcf)

            self.last_mode = "STOP_LOST"
            key = path_msg_key if path_msg_key in (
                "search_failed", "path_ended") else "bridge_stop"
            msg = MESSAGES.get(key, MESSAGES["safety_stop"])
            return self._pack(key, msg, tactile_res,
                              mode="STOP_LOST", conf=0.0)

        # ── 4. Тодорхойгүй ─────────────────────────────────────
        if STOP_ON_UNKNOWN:
            self.last_mode = "STOP_UNKNOWN"
            return self._pack("safety_stop", MESSAGES["safety_stop"],
                              tactile_res, mode="STOP_UNKNOWN", conf=0.0)

        out = tactile_res.copy()
        out["mode"] = "FALLBACK"
        return out

    # ── Private ─────────────────────────────────────────────────

    @staticmethod
    def _apply_gps_overlay(result: dict, gps_res):
        if not gps_res:
            return
        if gps_res.get("arrived"):
            result["message"]   = MESSAGES["gps_arrived"]
            result["color"]     = COLORS["gps_arrived"]
            result["arrow"]     = ARROWS["gps_arrived"]
            result["direction"] = "gps_arrived"
        elif (gps_res["key"] in ("gps_turn_left", "gps_turn_right") and
              gps_res["dist"] < 20):
            side = "ZUUN" if gps_res["key"] == "gps_turn_left" else "BARUUN"
            result["message"] += f" | GPS:{side} {gps_res['dist']}m"

    @staticmethod
    def _pack(key: str, msg: str, tactile_res: dict,
              mode: str = "?", conf: float = 1.0) -> dict:
        return {
            "direction": key,
            "conf":      float(conf),
            "offset":    tactile_res.get("offset", 0.0),
            "message":   msg,
            "color":     COLORS.get(key, (100, 100, 100)),
            "arrow":     ARROWS.get(key, "?"),
            "ratios":    tactile_res.get("ratios", (0, 0, 0)),
            "drift":     tactile_res.get("drift", 0.0),
            "cxs":       tactile_res.get("cxs", []),
            "pixels":    tactile_res.get("pixels", 0),
            "mode":      mode,
        }
