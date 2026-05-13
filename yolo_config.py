"""
YOLO Configuration Helper
Sets up safe directories for YOLO models on both local and Render servers
"""
import os
import sys
import tempfile
from pathlib import Path


def setup_yolo_environment():
    """
    Configure YOLO to work on both local machines and Render server.
    Must be called BEFORE importing YOLO from ultralytics.
    """
    
    # Determine writable cache directory
    # Priority: local project cache -> /tmp -> system temp
    
    # 1. Try project-local cache first (for local development)
    local_cache = Path(__file__).parent / ".yolo_cache"
    
    # 2. On Render, /tmp is writable; locally we prefer project cache
    if os.access("/tmp", os.W_OK):
        # We're on a Unix system with /tmp
        writable_cache = "/tmp/yolo_cache"
    else:
        # Windows or restricted environment
        writable_cache = str(local_cache)
    
    # Create cache directory if needed
    try:
        Path(writable_cache).mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CACHE"] = writable_cache
    except Exception as e:
        print(f"[WARN] Could not create cache at {writable_cache}: {e}")
        # Fallback to temp directory
        try:
            temp_dir = tempfile.gettempdir()
            yolo_tmp = os.path.join(temp_dir, "yolo_cache")
            Path(yolo_tmp).mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CACHE"] = yolo_tmp
        except Exception as e2:
            print(f"[ERROR] Could not set up YOLO cache: {e2}")
    
    # Configure config directory (for settings)
    # Use the same directory as cache to keep everything together
    yolo_cache = os.environ.get("YOLO_CACHE", writable_cache)
    config_dir = os.path.join(yolo_cache, "config")
    
    try:
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = config_dir
    except Exception as e:
        print(f"[WARN] Could not create config dir: {e}")
    
    # Disable verbose YOLO output
    os.environ["YOLO_VERBOSE"] = "False"
    
    # Suppress unnecessary logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
    
    print(f"[INFO] YOLO Cache: {os.environ.get('YOLO_CACHE')}")
    print(f"[INFO] YOLO Config: {os.environ.get('YOLO_CONFIG_DIR')}")


# Auto-configure on import
setup_yolo_environment()
