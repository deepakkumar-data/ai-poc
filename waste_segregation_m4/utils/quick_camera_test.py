#!/usr/bin/env python3
"""
Quick Camera Test - Interactive
Opens camera and shows a live preview to verify it works
"""

import cv2
import sys

def quick_test(camera_index: int = 0):
    """Quick test: open camera and show preview."""
    print(f"🔍 Testing camera {camera_index}...")
    print("Press 'q' to quit, 's' to save a test image\n")
    
    # Try different backends
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_QT, "QuickTime"),
    ]
    
    cap = None
    backend_name = None
    
    for backend_id, name in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    backend_name = name
                    print(f"✅ Camera opened using {name}")
                    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    break
                else:
                    cap.release()
                    cap = None
        except Exception as e:
            if cap:
                cap.release()
                cap = None
            continue
    
    if cap is None or not cap.isOpened():
        print(f"\n❌ Could not open camera {camera_index}")
        print("\nTroubleshooting:")
        print("1. Check camera permissions:")
        print("   System Settings → Privacy & Security → Camera")
        print("   Enable Terminal/IDE/Python")
        print("2. Try a different camera index:")
        print(f"   python {sys.argv[0]} 1")
        print("3. Close other apps using the camera")
        return False
    
    print("\n📹 Camera preview window opened")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save a test image")
    print("   - If you see video, your camera is working! ✅\n")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add text overlay
            cv2.putText(frame, f"Camera {camera_index} - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f"Camera {camera_index} Test - {backend_name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✅ Camera test completed successfully!")
                break
            elif key == ord('s'):
                filename = f"camera_test_{camera_index}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved test image: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Test Summary:")
        print(f"   - Camera {camera_index} opened successfully")
        print(f"   - Backend: {backend_name}")
        print(f"   - Frames captured: {frame_count}")
        print(f"\n✅ Your camera is ready for Live Camera mode!")
    
    return True

if __name__ == "__main__":
    camera_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    success = quick_test(camera_idx)
    sys.exit(0 if success else 1)
