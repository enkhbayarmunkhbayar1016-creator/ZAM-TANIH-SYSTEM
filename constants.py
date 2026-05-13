"""
Tactile Navigation v4 — Бүх тогтмол утгууд
"""

# ── Inference тохиргоо ─────────────────────────────────────────
CONF        = 0.25
IMGSZ       = 320
HIST_SIZE   = 6
VOICE_GAP   = 2.0
DRIFT_T     = 0.15

# ── Пиксел босго (тактил хавтны хэмжээ) ───────────────────────
PIX_NORMAL   = 2000
PIX_WARNING  = 1000
PIX_CRITICAL = 400
PIX_LOST     = 150

# ── Хайлтын горим ─────────────────────────────────────────────
WAIT_FRAMES   = 20
SEARCH_FRAMES = 90

# ── Саад илрүүлэлт ────────────────────────────────────────────
OBSTACLE_IMGSZ    = 320
OBSTACLE_CONF     = 0.40
OBSTACLE_INTERVAL = 5

# ── Walkable path параметрүүд ──────────────────────────────────
WALK_ROI_TOP      = 0.50
WALK_GRID_W       = 16
WALK_GRID_H       = 8
WALK_TEX_THRESH   = 250.0
WALK_EDGE_THRESH  = 0.18
WALK_MIN_CELLS    = 6
WALK_MIN_CONF     = 0.40
WALK_SAT_MAX      = 110
WALK_VAL_MIN      = 40

# ── Safe Bridge параметрүүд ────────────────────────────────────
BRIDGE_MAX_SEC      = 1.5
BRIDGE_MIN_FREE     = 0.55
BRIDGE_NO_OBSTACLE  = True
BRIDGE_MIN_CONF     = 0.50

# ── Аюулгүй ажиллагааны минимум ───────────────────────────────
SAFETY_MIN_CONF   = 0.30
STOP_ON_UNKNOWN   = True

# ── COCO аюултай ангилал ──────────────────────────────────────
COCO_DANGER = {
    0:  ("hun",        "high"),
    1:  ("dugui",      "high"),
    2:  ("mashin",     "high"),
    3:  ("motor",      "high"),
    5:  ("avtobus",    "high"),
    7:  ("achaa",      "high"),
    15: ("muur",       "low"),
    16: ("nokhoi",     "medium"),
    56: ("sandal",     "low"),
    57: ("buguudai",   "low"),
    58: ("modnii_sav", "low"),
    60: ("shiree",     "low"),
}

# ── Дэлгэц хэмжээ ─────────────────────────────────────────────
CAM_W, CAM_H = 640, 480
MAP_W, MAP_H = 320, 480

# ── Мэдэгдлүүд ────────────────────────────────────────────────
MESSAGES = {
    "straight":         "SHULUUN  | Zam urd baina",
    "straight_left":    "SHULUUN  | Arai zuun tiish",
    "straight_right":   "SHULUUN  | Arai baruun tiish",
    "turn_left":        "ZUUN     | Zam zuun tiish",
    "turn_right":       "BARUUN   | Zam baruun tiish",
    "intersection":     "UULZVAR  | Bolgoomojtoi",
    "path_warning":     "ANKHAAR  | Zam duusahad oirtloo",
    "path_ending":      "UDAASHRA | Zam udakhgui duusna",
    "path_ended":       "ZAM ALGA | Zam duuslaa - Zogsono uu",
    "search_start":     "HAIKH    | Zam haij baina",
    "search_left":      "HAIKH    | Kameraa zuun tiish",
    "search_right":     "HAIKH    | Kameraa baruun tiish",
    "search_found":     "OLDLOO   | Zam dahin oldloo",
    "search_failed":    "TUSLALTS | Tuslamj duudna uu",
    "danger_high":      "ZOGS     | URD OIRHON AYUL",
    "danger_medium":    "ANKHAAR  | URD BARTAA",
    "danger_low":       "BOLGOOM  | URD BARTAA",
    "danger_left":      "TOIRCH   | Baruun tiish",
    "danger_right":     "TOIRCH   | Zuun tiish",
    "gps_straight":     "GPS      | Shuluun yavna uu",
    "gps_turn_left":    "GPS      | {dist}m daraa ZUUN ergene",
    "gps_turn_right":   "GPS      | {dist}m daraa BARUUN ergene",
    "gps_arrived":      "GPS      | Ochih gazart hurlee!",
    "gps_off_route":    "GPS      | Zamaas garlaa",
    "no_path":          "ZAM ALGA | Havtan haragdahgui",
    "walk_straight":    "VISION   | Shuluun yav (havtangui)",
    "walk_left":        "VISION   | Zuun tiish yav (havtangui)",
    "walk_right":       "VISION   | Baruun tiish yav (havtangui)",
    "walk_unsafe":      "ZOGS     | Yavah zam haragdahgui",
    "bridge_continue":  "BRIDGE   | Shuluun urgeljluulne",
    "bridge_stop":      "ZOGS     | Havtan tasarsan - ayulgui bish",
    "safety_stop":      "ZOGS     | Itgel sul - ayulgui bish",
}

COLORS = {
    "straight":         (0, 220, 20),
    "straight_left":    (0, 200, 80),
    "straight_right":   (0, 200, 80),
    "turn_left":        (30, 200, 255),
    "turn_right":       (255, 160, 20),
    "intersection":     (180, 0, 255),
    "path_warning":     (0, 200, 200),
    "path_ending":      (0, 140, 255),
    "path_ended":       (0, 30, 255),
    "search_start":     (200, 200, 0),
    "search_left":      (200, 200, 0),
    "search_right":     (200, 200, 0),
    "search_found":     (0, 220, 20),
    "search_failed":    (0, 0, 200),
    "danger_high":      (0, 0, 255),
    "danger_medium":    (0, 100, 255),
    "danger_low":       (0, 200, 255),
    "danger_left":      (0, 100, 255),
    "danger_right":     (0, 100, 255),
    "gps_straight":     (100, 200, 100),
    "gps_turn_left":    (100, 200, 100),
    "gps_turn_right":   (100, 200, 100),
    "gps_arrived":      (0, 220, 20),
    "gps_off_route":    (0, 100, 200),
    "no_path":          (100, 100, 200),
    "walk_straight":    (180, 220, 80),
    "walk_left":        (180, 200, 100),
    "walk_right":       (180, 200, 100),
    "walk_unsafe":      (0, 0, 255),
    "bridge_continue":  (200, 180, 60),
    "bridge_stop":      (0, 30, 255),
    "safety_stop":      (0, 0, 220),
}

ARROWS = {
    "straight": " ^ ", "straight_left": "^< ", "straight_right": " >^",
    "turn_left": "<--", "turn_right": "-->", "intersection": "<+>",
    "path_warning": " ! ", "path_ending": "!!", "path_ended": "[X]",
    "search_start": "???", "search_left": "<? ", "search_right": " ?>",
    "search_found": " ^ ", "search_failed": "SOS",
    "danger_high": "[!]", "danger_medium": "[!]", "danger_low": " ! ",
    "danger_left": ">  ", "danger_right": "  <",
    "gps_straight": "GPS", "gps_turn_left": "G<-", "gps_turn_right": "->G",
    "gps_arrived": "[*]", "gps_off_route": "G?",
    "no_path": "[X]",
    "walk_straight": " ^ ", "walk_left": "<--", "walk_right": "-->",
    "walk_unsafe": "[X]",
    "bridge_continue": "...", "bridge_stop": "[X]",
    "safety_stop": "[X]",
}
