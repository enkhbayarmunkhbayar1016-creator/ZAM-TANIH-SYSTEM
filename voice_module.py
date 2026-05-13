"""
Дуут мэдэгдлийн модуль
"""
import threading
import time
from constants import VOICE_GAP

try:
    import pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False


class Voice:
    PRIORITY = {
        "danger_high": 100, "danger_medium": 90,
        "safety_stop": 95,  "bridge_stop": 95,
        "walk_unsafe": 85,
        "path_ended": 80,   "no_path": 80,
        "search_failed": 75, "gps_arrived": 70,
        "intersection": 60,
        "turn_left": 50,    "turn_right": 50,
        "gps_turn_left": 45, "gps_turn_right": 45,
        "path_warning": 40,  "path_ending": 35,
        "walk_straight": 25, "walk_left": 30, "walk_right": 30,
        "bridge_continue": 30,
        "straight": 20,
    }

    def __init__(self):
        self.engine   = None
        self.last_msg = ""
        self.last_t   = 0.0
        self.busy     = False
        self.enabled  = False
        if TTS_OK:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 145)
            except Exception:
                pass

    def say(self, msg: str, direction_key: str, force: bool = False):
        if not self.enabled:
            return
        now = time.time()
        if not force and msg == self.last_msg and now - self.last_t < VOICE_GAP:
            return
        if self.busy and self.PRIORITY.get(direction_key, 30) < 80:
            return
        self.last_msg = msg
        self.last_t   = now
        if self.engine:
            threading.Thread(target=self._speak, args=(msg,), daemon=True).start()

    def _speak(self, msg: str):
        self.busy = True
        try:
            self.engine.say(msg)
            self.engine.runAndWait()
        except Exception:
            pass
        finally:
            self.busy = False
