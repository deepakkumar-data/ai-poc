#!/bin/bash

# Camera Permissions Fix Script for macOS
# Helps diagnose and fix camera permission issues

echo "🔍 Camera Permissions Diagnostic Tool"
echo "======================================"
echo ""

# Check macOS version
echo "📱 macOS Version:"
sw_vers
echo ""

# Check if Terminal has camera access
echo "🔐 Checking camera permissions..."
echo ""

# Test camera access with Python
echo "Testing camera access..."
python3 << 'EOF'
import cv2
import sys

print("Testing camera access...")
print("")

# Try different camera indices
for i in range(5):
    print(f"Testing camera {i}...")
    
    # Try AVFoundation backend (macOS native)
    try:
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✅ Camera {i}: WORKING (AVFoundation)")
                print(f"     Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                sys.exit(0)
            else:
                cap.release()
        else:
            print(f"  ❌ Camera {i}: Not accessible")
    except Exception as e:
        print(f"  ⚠️  Camera {i}: Error - {e}")

print("")
print("❌ No working cameras found!")
print("")
print("SOLUTION:")
print("1. Open System Settings → Privacy & Security → Camera")
print("2. Enable Terminal (or your IDE)")
print("3. Restart Terminal/IDE")
print("4. Run this script again")
EOF

echo ""
echo "======================================"
echo "If no cameras work, follow these steps:"
echo ""
echo "1. Open System Settings"
echo "2. Go to Privacy & Security → Camera"
echo "3. Find and enable:"
echo "   - Terminal (if running from terminal)"
echo "   - Python (if available)"
echo "   - Your IDE (VS Code, PyCharm, etc.)"
echo ""
echo "4. Restart your terminal/IDE"
echo "5. Run this script again:"
echo "   bash utils/fix_camera_permissions.sh"
echo ""
