"""
Quick test script to verify YOLO setup works on both local and Render servers
Run this to diagnose any issues: python test_yolo_setup.py
"""
import os
import sys
from pathlib import Path

print("╔════════════════════════════════════════════════════════════╗")
print("║  YOLO Configuration Test                                   ║")
print("╚════════════════════════════════════════════════════════════╝\n")

# Step 1: Check yolo_config import
print("[1/5] Testing yolo_config import...")
try:
    import yolo_config
    print("     ✅ yolo_config imported successfully")
except Exception as e:
    print(f"     ❌ ERROR: {e}")
    sys.exit(1)

# Step 2: Check environment variables
print("\n[2/5] Checking environment variables...")
yolo_cache = os.environ.get("YOLO_CACHE", "NOT SET")
yolo_config_dir = os.environ.get("YOLO_CONFIG_DIR", "NOT SET")
print(f"     YOLO_CACHE: {yolo_cache}")
print(f"     YOLO_CONFIG_DIR: {yolo_config_dir}")

if yolo_cache == "NOT SET":
    print("     ⚠️  WARNING: YOLO_CACHE not set")
else:
    print("     ✅ Cache directory configured")

# Step 3: Check cache directory
print("\n[3/5] Checking cache directories...")
if yolo_cache != "NOT SET" and Path(yolo_cache).exists():
    print(f"     ✅ {yolo_cache} exists")
    # Check writability
    test_file = Path(yolo_cache) / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
        print(f"     ✅ {yolo_cache} is writable")
    except Exception as e:
        print(f"     ❌ {yolo_cache} is NOT writable: {e}")
else:
    print(f"     ⚠️  Cache directory not found/configured")

# Step 4: Check model files
print("\n[4/5] Checking model files...")
models = ["best.pt", "yolov8n.pt"]
for model in models:
    if Path(model).exists():
        size_mb = Path(model).stat().st_size / (1024 * 1024)
        print(f"     ✅ {model} ({size_mb:.1f} MB)")
    else:
        print(f"     ❌ {model} NOT FOUND")

# Step 5: Test YOLO import and loading
print("\n[5/5] Testing YOLO model loading...")
try:
    from ultralytics import YOLO
    print("     ✅ YOLO imported successfully")
    
    # Try loading a small model
    print("     Loading yolov8n.pt (this may take 30-60 seconds on first run)...")
    model = YOLO("yolov8n.pt")
    print("     ✅ Model loaded successfully!")
    print(f"     Model type: {type(model)}")
    
except Exception as e:
    print(f"     ❌ ERROR loading model: {e}")
    sys.exit(1)

print("\n╔════════════════════════════════════════════════════════════╗")
print("║  ✅ All checks passed! Ready for deployment               ║")
print("╚════════════════════════════════════════════════════════════╝\n")

print("Next steps:")
print("  1. Local: python app.py")
print("  2. Render: Deploy with start.sh (see RENDER_DEPLOYMENT.md)")
