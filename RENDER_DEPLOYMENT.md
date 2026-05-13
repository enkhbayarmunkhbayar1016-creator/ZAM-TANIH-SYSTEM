# Render Deployment Guide

## Issue Fixed
The warning about YOLO config directory not being writable on Render has been resolved.

## What Changed

### New Files
- **`yolo_config.py`** - Automatically configures YOLO before model loading
  - Sets writable cache directory to `/tmp/yolo_cache`
  - Creates necessary directories automatically
  - Suppresses verbose output

- **`start.sh`** - Deployment startup script for Render
  - Verifies model files exist
  - Creates cache directories
  - Sets environment variables
  - Starts gunicorn server

- **`.env.example`** - Configuration template

### Modified Files
- **`app.py`** - Imports `yolo_config` before YOLO
- **`navigate_v4.py`** - Imports `yolo_config` before YOLO

## Deployment on Render

### Step 1: Update render.yaml
```yaml
services:
  - type: web
    name: tactile-nav
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: bash start.sh
    envVars:
      - key: PORT
        value: 5000
      - key: TACTILE_MODEL
        value: best.pt
      - key: OBSTACLE_MODEL
        value: yolov8n.pt
```

### Step 2: Ensure Model Files in Repo
Make sure these files are committed to your repository:
- `best.pt` (custom tactile model)
- `yolov8n.pt` (obstacle detection model)

### Step 3: Deploy
Push to your repository - Render will automatically:
1. Install dependencies
2. Run `start.sh` which sets up cache directories
3. Start the Flask app

## Local Testing

### Test Locally First
```bash
# Make start.sh executable
chmod +x start.sh

# Run locally
python app.py
# OR
bash start.sh
```

### Environment Variables
The `yolo_config.py` module automatically handles:
- ✅ Creates `/tmp/yolo_cache` on Render
- ✅ Falls back to local cache on development machines
- ✅ Sets proper permissions
- ✅ Suppresses YOLO warnings

No manual environment setup needed!

## Troubleshooting

### If models aren't loading:
1. Check that `best.pt` and `yolov8n.pt` exist
2. Run `python -c "import yolo_config; from ultralytics import YOLO"` 
3. Check Render logs for permission errors

### If cache errors persist:
1. The app will auto-recover using different cache directories
2. Check `/tmp/yolo_cache` exists and is writable
3. View logs: Render Dashboard → Your App → Logs

## Performance Notes
- Models are loaded once on app start (no reload per request)
- Cache persists during dyno lifetime (can be cleared with restart)
- On Render free tier, expect longer first load (~20-30 seconds)
