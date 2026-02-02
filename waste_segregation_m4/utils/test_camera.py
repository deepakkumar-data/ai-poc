#!/usr/bin/env python3
"""
Camera Testing Script
Tests available cameras - cross-platform support (macOS, Windows, Linux)
"""

import cv2
import sys
import platform

def test_cameras(max_index: int = 5):
    """Test cameras from index 0 to max_index."""
    print("🔍 Testing available cameras...\n")
    
    available_cameras = []
    _system = platform.system()
    
    # Select appropriate backends for the platform
    if _system == "Darwin":  # macOS
        backends = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS native)"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_QT, "QuickTime"),
        ]
    elif _system == "Windows":  # Windows
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (Windows native)"),
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
        ]
    else:  # Linux and others
        backends = [
            (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
            (cv2.CAP_ANY, "Default"),
        ]
    
    for i in range(max_index):
        camera_found = False
        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(i, backend_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✅ Camera {i}: Available and working")
                        print(f"   Backend: {backend_name}")
                        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        print(f"   Channels: {frame.shape[2] if len(frame.shape) > 2 else 1}")
                        available_cameras.append((i, backend_name))
                        camera_found = True
                        cap.release()
                        break
                    else:
                        cap.release()
            except Exception as e:
                if cap.isOpened():
                    cap.release()
                continue
        
        if not camera_found:
            print(f"❌ Camera {i}: Not available")
    
    print("\n" + "=" * 60)
    if available_cameras:
        print(f"✅ Found {len(available_cameras)} working camera(s):")
        for idx, backend in available_cameras:
            print(f"   - Camera {idx} (use backend: {backend})")
        print(f"\n💡 Recommended: Use camera index {available_cameras[0][0]}")
    else:
        print("❌ No cameras found!")
        print("\nTroubleshooting:")
        if _system == "Darwin":  # macOS
            print("1. Check camera permissions:")
            print("   System Settings → Privacy & Security → Camera")
            print("   Enable Terminal/IDE/Python")
        elif _system == "Windows":  # Windows
            print("1. Check camera permissions:")
            print("   Settings → Privacy → Camera")
            print("   Enable camera access for your application")
        else:  # Linux
            print("1. Check camera permissions:")
            print("   Ensure your user has access to /dev/video* devices")
        print("2. Make sure no other app is using the camera")
        print("3. Check if camera is connected (for external cameras)")
        sys.exit(1)
    
    return available_cameras

if __name__ == "__main__":
    max_cameras = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    test_cameras(max_cameras)
