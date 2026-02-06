"""
Conveyor Belt Engine Module
Handles live webcam feeds and video files for waste classification
with ROI masking, trigger line detection, and temporal synchronization
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque
import time
from classifier import WasteClassifier

# Camera backends - cross-platform support
import platform
_system = platform.system()

if _system == "Darwin":  # macOS
    CAMERA_BACKENDS = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS native)"),
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_QT, "QuickTime"),
    ]
elif _system == "Windows":  # Windows
    CAMERA_BACKENDS = [
        (cv2.CAP_DSHOW, "DirectShow (Windows native)"),
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_MSMF, "Microsoft Media Foundation"),
    ]
else:  # Linux and others
    CAMERA_BACKENDS = [
        (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
        (cv2.CAP_ANY, "Default"),
    ]


class ConveyorEngine:
    """
    Conveyor belt processing engine with ROI, trigger line, and temporal sync.
    """
    
    def __init__(
        self,
        classifier: WasteClassifier,
        roi: Optional[Tuple[int, int, int, int]] = None,
        trigger_line_x: Optional[int] = None,
        max_tracked_objects: int = 10,
        classification_cooldown: float = 2.0
    ):
        """
        Initialize the Conveyor Engine.
        
        Args:
            classifier: WasteClassifier instance for inference
            roi: Region of Interest as (x, y, width, height). If None, uses full frame
            trigger_line_x: X-coordinate of vertical trigger line. If None, uses center
            max_tracked_objects: Maximum number of objects to track simultaneously
            classification_cooldown: Minimum seconds between classifications of same object
        """
        self.classifier = classifier
        self.roi = roi
        self.trigger_line_x = trigger_line_x
        self.max_tracked_objects = max_tracked_objects
        self.classification_cooldown = classification_cooldown
        
        # Background subtractor for object detection
        # Lower varThreshold for better sensitivity (default 16, we use 25)
        # Lower history for faster adaptation (300 frames ~10 seconds at 30fps)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,  # Lower threshold = more sensitive (was 50)
            detectShadows=True
        )
        
        # Track frames processed for background learning
        self.frames_processed = 0
        self.background_learning_frames = 30  # Learn background for first 30 frames
        
        # Temporal sync: Track objects and their classification status
        self.tracked_objects = {}  # {object_id: {centroid, last_classified_time, label, confidence}}
        self.next_object_id = 0
        self.classification_queue = deque(maxlen=max_tracked_objects)
        
        # Frame dimensions (set when processing starts)
        self.frame_width = None
        self.frame_height = None
        
        # Motion detection
        self.previous_frame = None
        self.motion_threshold = 0.01  # 1% of ROI pixels must change for motion (was 0.02)
        self.motion_detected = False
        self.min_motion_area = 200  # Minimum area of motion pixels (was 500)
        
        # Visual overlay settings
        self.roi_color = (0, 255, 0)  # Green
        self.trigger_line_color = (255, 0, 0)  # Blue
        self.text_color = (255, 255, 255)  # White
        self.box_color = (0, 255, 255)  # Yellow for detected objects
        
    def _setup_roi_and_trigger(self, frame: np.ndarray):
        """Setup ROI and trigger line based on frame dimensions."""
        if self.frame_width is None or self.frame_height is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # Set default ROI if not provided (90% of frame, centered)
            # Increased from 80% to 90% to capture more of the frame for detection
            if self.roi is None:
                roi_width = int(self.frame_width * 0.9)
                roi_height = int(self.frame_height * 0.9)
                roi_x = (self.frame_width - roi_width) // 2
                roi_y = (self.frame_height - roi_height) // 2
                self.roi = (roi_x, roi_y, roi_width, roi_height)
            
            # Set default trigger line to center if not provided
            if self.trigger_line_x is None:
                self.trigger_line_x = self.frame_width // 2
    
    def _get_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a mask for the ROI region."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x, y, w, h = self.roi
        mask[y:y+h, x:x+w] = 255
        return mask
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness=1, dash_length=10):
        """Draw a dashed rectangle on the image."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Top and bottom lines
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left and right lines
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _detect_objects(self, frame: np.ndarray, roi_mask: np.ndarray) -> List[Dict]:
        """
        Detect objects using background subtraction.
        
        Returns:
            List of detected objects with bounding boxes and centroids
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply ROI mask
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        # Increased minimum area to ensure we detect complete objects, not noise
        # But still lower than original to catch smaller objects
        min_area = 400  # Minimum object area to filter noise (balanced: not too small, not too large)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    objects.append({
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'area': area
                    })
        
        return objects
    
    def _assign_object_ids(self, detected_objects: List[Dict]) -> List[Dict]:
        """
        Assign IDs to detected objects using simple centroid tracking.
        Matches objects to existing tracked objects based on centroid proximity.
        IMPORTANT: Prevents the same physical object from getting multiple IDs.
        """
        current_time = time.time()
        matched_ids = set()
        updated_objects = []
        
        for obj in detected_objects:
            cx, cy = obj['centroid']
            best_match_id = None
            min_distance = float('inf')
            
            # Find closest existing tracked object
            for obj_id, tracked_data in self.tracked_objects.items():
                if obj_id in matched_ids:
                    continue
                
                tracked_cx, tracked_cy = tracked_data['centroid']
                distance = np.sqrt((cx - tracked_cx)**2 + (cy - tracked_cy)**2)
                
                # Increased threshold for better tracking (especially for classified objects)
                # If object was already classified, use larger threshold to prevent new ID
                max_match_distance = 150 if not tracked_data.get('has_crossed', False) else 200
                
                if distance < max_match_distance and distance < min_distance:
                    min_distance = distance
                    best_match_id = obj_id
            
            # Assign ID
            if best_match_id is not None:
                # Update existing object
                obj['id'] = best_match_id
                self.tracked_objects[best_match_id]['centroid'] = (cx, cy)
                matched_ids.add(best_match_id)
            else:
                # Create new object
                obj['id'] = self.next_object_id
                self.tracked_objects[self.next_object_id] = {
                    'centroid': (cx, cy),
                    'last_classified_time': 0,
                    'label': None,
                    'confidence': None,
                    'has_crossed': False
                }
                self.next_object_id += 1
            
            updated_objects.append(obj)
        
        # Remove objects that are no longer detected (cleanup old tracks)
        # IMPORTANT: Keep classified objects longer to prevent re-counting
        active_ids = {obj['id'] for obj in updated_objects}
        ids_to_remove = []
        for obj_id in self.tracked_objects.keys():
            if obj_id not in active_ids:
                # If object was classified, keep it for 30 seconds to prevent re-counting
                if self.tracked_objects[obj_id].get('has_crossed', False):
                    time_since_classified = current_time - self.tracked_objects[obj_id].get('last_classified_time', 0)
                    if time_since_classified > 30.0:  # Remove after 30 seconds
                        ids_to_remove.append(obj_id)
                else:
                    # Unclassified objects can be removed immediately
                    ids_to_remove.append(obj_id)
        
        for obj_id in ids_to_remove:
            del self.tracked_objects[obj_id]
        
        return updated_objects
    
    def _detect_motion(self, frame: np.ndarray, roi_mask: np.ndarray) -> bool:
        """
        Detect if there's motion in the frame (conveyor is moving).
        
        Args:
            frame: Current frame
            roi_mask: ROI mask to focus on conveyor area
            
        Returns:
            True if motion detected, False otherwise
        """
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return False
        
        # Convert to grayscale for comparison
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(gray_current, gray_previous)
        
        # Apply ROI mask
        frame_diff = cv2.bitwise_and(frame_diff, roi_mask)
        
        # Threshold to get motion pixels
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate motion area
        motion_area = cv2.countNonZero(motion_mask)
        roi_area = cv2.countNonZero(roi_mask)
        
        # Check if motion is significant
        if roi_area > 0:
            motion_ratio = motion_area / roi_area
            # Motion detected if ratio exceeds threshold and area is significant
            self.motion_detected = (motion_ratio >= self.motion_threshold and 
                                   motion_area >= self.min_motion_area)
        else:
            self.motion_detected = False
        
        # Update previous frame
        self.previous_frame = frame.copy()
        
        return self.motion_detected
    
    def _check_trigger_line_crossing(self, obj: Dict) -> bool:
        """
        Check if object has crossed the trigger line and should be classified.
        Implements temporal sync to prevent duplicate classifications.
        Can classify with or without motion (for static objects after background learning).
        """
        obj_id = obj['id']
        cx, cy = obj['centroid']
        tracked_data = self.tracked_objects[obj_id]
        current_time = time.time()
        
        # Check if object has already crossed and been classified
        # IMPORTANT: Once an object is classified, it should NEVER be classified again
        # even if it moves or motion is detected again
        if tracked_data.get('has_crossed', False):
            return False
        
        # Allow classification if:
        # 1. Motion is detected (conveyor moving), OR
        # 2. Background is ready (learned enough) - allows static object detection
        bg_ready = self.frames_processed >= self.background_learning_frames
        can_proceed = self.motion_detected or bg_ready
        
        if not can_proceed:
            return False
        
        # Check if centroid has crossed the trigger line
        # We need to track previous position to detect crossing
        # For simplicity, we'll classify when object is at or past the trigger line
        # and hasn't been classified recently
        
        # Check if object is at/past trigger line
        has_crossed_line = cx >= self.trigger_line_x
        
        # Check cooldown period (extra safety)
        time_since_last = current_time - tracked_data.get('last_classified_time', 0)
        can_classify = time_since_last >= self.classification_cooldown
        
        # Only classify if:
        # 1. Object has crossed the trigger line
        # 2. Cooldown period has passed
        # 3. Object hasn't been classified before (has_crossed check above)
        if has_crossed_line and can_classify:
            # Mark as crossed IMMEDIATELY to prevent any duplicate classifications
            tracked_data['has_crossed'] = True
            tracked_data['last_classified_time'] = current_time
            return True
        
        return False
    
    def _classify_object(self, frame: np.ndarray, obj: Dict) -> Tuple[str, float]:
        """
        Classify an object using the waste classifier.
        Extracts ROI around the object and runs inference.
        Uses larger padding to capture more context for better accuracy.
        """
        x, y, w, h = obj['bbox']
        
        # Expand bounding box significantly for better classification accuracy
        # Use percentage-based padding to ensure we capture enough context
        # For small objects, use larger padding; for large objects, use proportional padding
        padding_ratio = 0.3  # 30% padding on all sides
        padding_x = max(30, int(w * padding_ratio))  # At least 30 pixels, or 30% of width
        padding_y = max(30, int(h * padding_ratio))  # At least 30 pixels, or 30% of height
        
        # Ensure minimum size for classification (models work better with larger images)
        min_size = 100  # Minimum dimension for classification
        if w < min_size:
            padding_x = max(padding_x, (min_size - w) // 2 + 10)
        if h < min_size:
            padding_y = max(padding_y, (min_size - h) // 2 + 10)
        
        # Calculate expanded bounding box
        x_expanded = max(0, x - padding_x)
        y_expanded = max(0, y - padding_y)
        w_expanded = min(frame.shape[1] - x_expanded, w + 2 * padding_x)
        h_expanded = min(frame.shape[0] - y_expanded, h + 2 * padding_y)
        
        # Extract object region with context
        object_roi = frame[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
        
        if object_roi.size == 0 or object_roi.shape[0] < 50 or object_roi.shape[1] < 50:
            print(f"⚠️  Object ROI too small for classification: {object_roi.shape if object_roi.size > 0 else 'empty'}")
            return "unknown", 0.0
        
        # Resize if too small (but maintain aspect ratio)
        # Most models expect at least 224x224 or similar
        target_min_size = 224
        if min(object_roi.shape[0], object_roi.shape[1]) < target_min_size:
            scale = target_min_size / min(object_roi.shape[0], object_roi.shape[1])
            new_w = int(object_roi.shape[1] * scale)
            new_h = int(object_roi.shape[0] * scale)
            object_roi = cv2.resize(object_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"📏 Resized object ROI from {object_roi.shape[:2]} to {new_h}x{new_w} for better classification")
        
        # Classify
        try:
            label, confidence = self.classifier.predict(object_roi)
            print(f"🔍 Classified with ROI size: {object_roi.shape[1]}x{object_roi.shape[0]} (original bbox: {w}x{h})")
            return label, confidence
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return "unknown", 0.0
    
    def _draw_overlays(
        self, 
        frame: np.ndarray, 
        objects: List[Dict],
        show_bg_mask: bool = False
    ) -> np.ndarray:
        """
        Draw visual overlays on the frame:
        - ROI box
        - Trigger line
        - Object bounding boxes
        - Classification results
        - Motion status
        """
        overlay = frame.copy()
        
        # Draw ROI box
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.roi_color, 2)
            cv2.putText(overlay, "ROI", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.roi_color, 2)
        
        # Draw trigger line
        if self.trigger_line_x:
            cv2.line(overlay, 
                    (self.trigger_line_x, 0), 
                    (self.trigger_line_x, self.frame_height),
                    self.trigger_line_color, 2)
            cv2.putText(overlay, "TRIGGER", 
                       (self.trigger_line_x + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.trigger_line_color, 2)
        
        # Draw motion status
        motion_color = (0, 255, 0) if self.motion_detected else (0, 0, 255)
        motion_text = "MOTION: YES" if self.motion_detected else "MOTION: NO"
        cv2.putText(overlay, motion_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, motion_color, 2)
        
        # Draw objects and classifications
        for obj in objects:
            x, y, w, h = obj['bbox']
            cx, cy = obj['centroid']
            obj_id = obj['id']
            tracked_data = self.tracked_objects.get(obj_id, {})
            
            # Draw bounding box
            color = self.box_color
            if tracked_data.get('has_crossed', False):
                color = (0, 255, 0)  # Green for classified objects
            
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw expanded classification region (dashed line) if object is being classified
            if tracked_data.get('label') or (cx >= self.trigger_line_x if self.trigger_line_x else False):
                # Calculate expanded region (same as in _classify_object)
                padding_ratio = 0.3
                padding_x = max(30, int(w * padding_ratio))
                padding_y = max(30, int(h * padding_ratio))
                min_size = 100
                if w < min_size:
                    padding_x = max(padding_x, (min_size - w) // 2 + 10)
                if h < min_size:
                    padding_y = max(padding_y, (min_size - h) // 2 + 10)
                
                x_exp = max(0, x - padding_x)
                y_exp = max(0, y - padding_y)
                w_exp = min(overlay.shape[1] - x_exp, w + 2 * padding_x)
                h_exp = min(overlay.shape[0] - y_exp, h + 2 * padding_y)
                
                # Draw expanded region with dashed line (yellow)
                self._draw_dashed_rectangle(overlay, (x_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), (0, 255, 255), 2)
            
            # Draw centroid
            cv2.circle(overlay, (cx, cy), 5, color, -1)
            
            # Draw object ID
            cv2.putText(overlay, f"ID:{obj_id}", (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw classification result if available
            if tracked_data.get('label'):
                label = tracked_data['label']
                confidence = tracked_data.get('confidence', 0.0)
                text = f"{label}: {confidence:.1%}"
                
                # Background for text readability
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(overlay, 
                            (x, y - text_height - 5),
                            (x + text_width, y),
                            (0, 0, 0), -1)
                
                cv2.putText(overlay, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
        
        # Blend overlay with original frame (semi-transparent)
        result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add background mask visualization if requested
        if show_bg_mask:
            fg_mask = self.bg_subtractor.apply(frame)
            roi_mask = self._get_roi_mask(frame)
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
            fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            result = np.hstack([result, fg_mask_colored])
        
        return result
    
    def process_frame(self, frame: np.ndarray, show_bg_mask: bool = False) -> np.ndarray:
        """
        Process a single frame through the conveyor engine.
        Only classifies objects when motion is detected (conveyor is moving).
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            show_bg_mask: If True, shows background subtraction mask alongside frame
            
        Returns:
            Processed frame with overlays
        """
        # Setup ROI and trigger line on first frame
        self._setup_roi_and_trigger(frame)
        
        # Get ROI mask
        roi_mask = self._get_roi_mask(frame)
        
        # Increment frame counter for background learning
        self.frames_processed += 1
        
        # Detect motion first (conveyor must be moving)
        motion_detected = self._detect_motion(frame, roi_mask)
        
        # Detect objects
        detected_objects = self._detect_objects(frame, roi_mask)
        
        # Assign IDs to objects
        tracked_objects = self._assign_object_ids(detected_objects)
        
        # Debug: Print detection status periodically
        if self.frames_processed % 30 == 0:  # Every 30 frames
            bg_ready = self.frames_processed >= self.background_learning_frames
            print(f"🔍 Detection Status: {len(detected_objects)} objects detected | "
                  f"Motion: {'Yes' if motion_detected else 'No'} | "
                  f"Background: {'Ready' if bg_ready else f'Learning ({self.frames_processed}/{self.background_learning_frames})'}")
        
        # Only check for trigger line crossings and classify if motion is detected
        # OR if background is ready and we have objects (allow static object detection after learning)
        bg_ready = self.frames_processed >= self.background_learning_frames
        can_classify = motion_detected or (bg_ready and len(tracked_objects) > 0)
        
        if can_classify:
            for obj in tracked_objects:
                if self._check_trigger_line_crossing(obj):
                    # Classify object
                    label, confidence = self._classify_object(frame, obj)
                    
                    # Store classification result
                    obj_id = obj['id']
                    self.tracked_objects[obj_id]['label'] = label
                    self.tracked_objects[obj_id]['confidence'] = confidence
                    
                    motion_status = "Yes" if motion_detected else "No (Static)"
                    print(f"✅ Classified Object {obj_id}: {label} ({confidence:.2%}) [Motion: {motion_status}]")
        elif len(tracked_objects) > 0 and not bg_ready:
            # Objects detected but background still learning
            if self.frames_processed % 30 == 0:
                print(f"⏳ Background learning in progress... ({self.frames_processed}/{self.background_learning_frames} frames)")
        
        # Draw overlays
        result_frame = self._draw_overlays(frame, tracked_objects, show_bg_mask)
        
        # Add detection status text to frame
        status_y = 60
        if not bg_ready:
            cv2.putText(result_frame, f"Background Learning: {self.frames_processed}/{self.background_learning_frames}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif len(detected_objects) == 0:
            cv2.putText(result_frame, "No objects detected - Adjust ROI or check lighting",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(result_frame, f"Objects detected: {len(detected_objects)}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result_frame
    
    def process_video_file(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        show_bg_mask: bool = False,
        fps: Optional[int] = None
    ):
        """
        Process a video file through the conveyor engine.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video. If None, displays video only
            show_bg_mask: If True, shows background subtraction mask
            fps: Output FPS. If None, uses input video FPS
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_fps = fps if fps else input_fps
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        frame_count = 0
        
        print(f"📹 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}, FPS: {input_fps}, Frames: {total_frames}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame, show_bg_mask)
                
                # Write or display frame
                if writer:
                    writer.write(processed_frame)
                else:
                    cv2.imshow('Conveyor Engine', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"✅ Video saved to: {output_path}")
            cv2.destroyAllWindows()
    
    def process_webcam(
        self, 
        camera_index: int = 0,
        show_bg_mask: bool = False
    ):
        """
        Process live webcam feed through the conveyor engine.
        
        Args:
            camera_index: Camera device index (default: 0)
            show_bg_mask: If True, shows background subtraction mask
        """
        cap = None
        for backend_id, backend_name in CAMERA_BACKENDS:
            try:
                cap = cv2.VideoCapture(camera_index, backend_id)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret:
                        print(f"✅ Camera {camera_index} opened using {backend_name}")
                        break
                    else:
                        cap.release()
                        cap = None
            except Exception:
                if cap:
                    cap.release()
                    cap = None
                continue
        
        if cap is None or not cap.isOpened():
            raise ValueError(
                f"Could not open camera {camera_index}\n\n"
                "Troubleshooting:\n"
                "1. Check camera permissions: System Settings → Privacy & Security → Camera\n"
                "2. Make sure no other app is using the camera\n"
                "3. Try a different camera index (1, 2, 3, etc.)\n"
                "4. Run: python utils/test_camera.py to find available cameras"
            )
        
        print("📹 Starting webcam feed. Press 'q' to quit, 'r' to reset background")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Failed to read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame, show_bg_mask)
                
                # Display frame
                cv2.imshow('Conveyor Engine - Webcam', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset background subtractor
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500,
                        varThreshold=50,
                        detectShadows=True
                    )
                    print("🔄 Background model reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def reset(self):
        """Reset the engine state (clear tracked objects, reset background model)."""
        self.tracked_objects = {}
        self.next_object_id = 0
        self.classification_queue.clear()
        self.previous_frame = None
        self.motion_detected = False
        self.frames_processed = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,  # Lower threshold = more sensitive
            detectShadows=True
        )
        print("🔄 Conveyor engine reset")


# Example usage
if __name__ == "__main__":
    from classifier import WasteClassifier
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = WasteClassifier(force_mps=True)
    
    # Initialize conveyor engine
    print("Initializing conveyor engine...")
    engine = ConveyorEngine(
        classifier=classifier,
        roi=None,  # Will use default (80% of frame, centered)
        trigger_line_x=None,  # Will use center of frame
        classification_cooldown=2.0
    )
    
    # Process webcam
    # engine.process_webcam(camera_index=0)
    
    # Or process video file
    # engine.process_video_file("input_video.mp4", "output_video.mp4")
    
    print("✅ Conveyor engine ready!")
    print("   Use engine.process_webcam() for live feed")
    print("   Use engine.process_video_file() for video files")
