"""
Зам алдагдах менежер
"""
import numpy as np
from collections import deque
from constants import (
    PIX_NORMAL, PIX_WARNING, PIX_CRITICAL, PIX_LOST,
    WAIT_FRAMES,
)


class PathLostManager:
    STATES = {"NORMAL", "WARNING", "ENDING", "LOST", "SEARCH", "FAILED"}

    def __init__(self):
        self.mode         = "NORMAL"
        self.lost_frames  = 0
        self.search_count = 0
        self.pixel_hist   = deque(maxlen=10)

    def update(self, pixels: int):
        self.pixel_hist.append(pixels)
        avg = float(np.mean(self.pixel_hist))

        if self.mode == "NORMAL":
            if avg > PIX_CRITICAL:
                if avg < PIX_WARNING:
                    self.mode = "WARNING"
                    return "WARNING", "path_warning"
                return "NORMAL", None
            elif avg > PIX_LOST:
                self.mode = "ENDING"
                return "ENDING", "path_ending"
            else:
                self.mode = "LOST"
                self.lost_frames = 0
                return "LOST", "path_ended"

        elif self.mode == "WARNING":
            if avg > PIX_NORMAL:
                self.mode = "NORMAL"
                return "NORMAL", None
            elif avg < PIX_LOST:
                self.mode = "LOST"
                self.lost_frames = 0
                return "LOST", "path_ended"
            return "WARNING", "path_warning"

        elif self.mode == "ENDING":
            if avg > PIX_WARNING:
                self.mode = "NORMAL"
                return "NORMAL", None
            elif avg < PIX_LOST:
                self.mode = "LOST"
                self.lost_frames = 0
                return "LOST", "path_ended"
            return "ENDING", "path_ending"

        elif self.mode == "LOST":
            if avg > PIX_CRITICAL:
                self.mode = "NORMAL"
                return "FOUND", "search_found"
            self.lost_frames += 1
            if self.lost_frames < WAIT_FRAMES:
                return "LOST", "path_ended"
            self.mode = "SEARCH"
            self.search_count = 0
            return "SEARCH", "search_start"

        elif self.mode == "SEARCH":
            if avg > PIX_CRITICAL:
                self.mode = "NORMAL"
                return "FOUND", "search_found"
            self.search_count += 1
            if self.search_count < 30:
                return "SEARCH", "search_start"
            elif self.search_count < 60:
                return "SEARCH", "search_left"
            elif self.search_count < 90:
                return "SEARCH", "search_right"
            else:
                self.mode = "FAILED"
                return "FAILED", "search_failed"

        elif self.mode == "FAILED":
            if avg > PIX_CRITICAL:
                self.mode = "NORMAL"
                return "FOUND", "search_found"
            return "FAILED", "search_failed"

        return "NORMAL", None

    def reset(self):
        self.mode         = "NORMAL"
        self.lost_frames  = 0
        self.search_count = 0
        self.pixel_hist.clear()
