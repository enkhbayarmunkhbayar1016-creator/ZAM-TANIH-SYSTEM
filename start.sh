#!/bin/bash
# Render deployment startup script
# Ensures YOLO cache is properly configured before app starts

echo "═════════════════════════════════════════════════════════════"
echo "🚀 Starting Tactile Navigation Server"
echo "═════════════════════════════════════════════════════════════"

# Verify models exist
if [ ! -f "best.pt" ]; then
    echo "❌ ERROR: best.pt not found in current directory"
    exit 1
fi

if [ ! -f "yolov8n.pt" ]; then
    echo "❌ ERROR: yolov8n.pt not found in current directory"
    exit 1
fi

echo "✅ Models found"

# Create cache directory
mkdir -p /tmp/yolo_cache
mkdir -p /tmp/yolo_cache/config

echo "✅ Cache directories created"

# Export environment variables
export YOLO_CACHE=/tmp/yolo_cache
export YOLO_CONFIG_DIR=/tmp/yolo_cache/config
export YOLO_VERBOSE=False
export TF_CPP_MIN_LOG_LEVEL=3

# Get the port - Render provides PORT, fallback to 5000
PORT=${PORT:-5000}

echo "✅ Environment configured"
echo "   YOLO_CACHE: $YOLO_CACHE"
echo "   YOLO_CONFIG_DIR: $YOLO_CONFIG_DIR"
echo "   PORT: $PORT"

# Start the Flask app with gunicorn
# Use workers=1 to minimize memory usage on free tier
# bind to all interfaces on the provided port
echo ""
echo "🔄 Starting Flask app on port $PORT..."
exec gunicorn --bind "0.0.0.0:$PORT" --workers 1 --worker-class sync --timeout 120 --access-logfile - --error-logfile - app:app
