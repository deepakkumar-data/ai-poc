"""
Streamlit Application for Waste Segregation System
Client prototype with live camera and video upload support
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os
from typing import Optional, Tuple, Dict
from collections import defaultdict
import traceback

from classifier import WasteClassifier
from conveyor_engine import ConveyorEngine
from constants import (
    RECYCLABLES,
    GENERAL_WASTE,
    DEFAULT_MODEL_KEY,
    DEFAULT_CLASSIFICATION_COOLDOWN,
    DEFAULT_FPS,
    MIN_FPS,
    MAX_FPS,
    RERUN_INTERVAL,
    DEFAULT_ROI_X,
    DEFAULT_ROI_Y,
    DEFAULT_ROI_WIDTH,
    DEFAULT_ROI_HEIGHT,
    MAX_ROI_X,
    MAX_ROI_Y,
    MODEL_NAME
)

# Page configuration
st.set_page_config(
    page_title="Waste Segregation System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Camera backends for macOS compatibility
CAMERA_BACKENDS = [
    (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS native)"),
    (cv2.CAP_ANY, "Default"),
    (cv2.CAP_QT, "QuickTime"),
]

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'classifier': None,
        'engine': None,
        'current_label': None,
        'current_confidence': 0.0,
        'processing_active': False,
        'camera': None,
        'camera_opened_message_shown': False,
        'camera_frame_count': 0,
        'camera_last_update': 0,
        'classification_history': [],  # List of {timestamp, label, confidence, image}
        'min_confidence_threshold': 0.5,  # Minimum confidence to show in sidebar
        'last_display_frame': None,  # Buffer for smooth display updates
        'session_stats': {  # Statistics for current session
            'total_items': 0,
            'recyclables_count': 0,
            'general_waste_count': 0,
            'items_by_type': defaultdict(int),
            'counted_object_ids': set(),  # Track which object IDs have been counted
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


@st.cache_resource
def load_classifier() -> WasteClassifier:
    """Load and cache the waste classifier."""
    with st.spinner("Loading waste classification model..."):
        classifier = WasteClassifier(
            model_key=DEFAULT_MODEL_KEY,
            force_mps=True
        )
    return classifier


def get_segregation_info(label: str) -> Tuple[str, str, str]:
    """
    Get segregation information for a label.
    
    Args:
        label: Classification label
        
    Returns:
        Tuple of (category_type, color, message)
    """
    label_lower = label.lower()
    
    if any(recyclable in label_lower for recyclable in RECYCLABLES):
        return "recyclable", "#00FF00", "♻️ RECYCLABLE"
    elif any(waste in label_lower for waste in GENERAL_WASTE):
        return "general_waste", "#FFA500", "🗑️ GENERAL WASTE"
    else:
        return "unknown", "#808080", "❓ UNKNOWN"


def get_roi_settings() -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
    """
    Get ROI and trigger line settings from sidebar.
    
    Returns:
        Tuple of (roi, trigger_line_x)
    """
    st.sidebar.subheader("Region of Interest (ROI)")
    use_custom_roi = st.sidebar.checkbox("Use Custom ROI", value=False, key="roi_checkbox")
    
    roi = None
    if use_custom_roi:
        roi_x = st.sidebar.slider("ROI X", 0, MAX_ROI_X, DEFAULT_ROI_X, key="roi_x")
        roi_y = st.sidebar.slider("ROI Y", 0, MAX_ROI_Y, DEFAULT_ROI_Y, key="roi_y")
        roi_w = st.sidebar.slider("ROI Width", 100, MAX_ROI_X, DEFAULT_ROI_WIDTH, key="roi_w")
        roi_h = st.sidebar.slider("ROI Height", 100, MAX_ROI_Y, DEFAULT_ROI_HEIGHT, key="roi_h")
        roi = (roi_x, roi_y, roi_w, roi_h)
    
    st.sidebar.subheader("Trigger Line")
    use_custom_trigger = st.sidebar.checkbox("Use Custom Trigger Line", value=False, key="trigger_checkbox")
    
    trigger_x = None
    if use_custom_trigger:
        trigger_x = st.sidebar.slider("Trigger Line X", 0, MAX_ROI_X, MAX_ROI_X // 2, key="trigger_x")
    
    return roi, trigger_x


def sidebar() -> Dict:
    """Create the sidebar with mode selection and settings."""
    # Professional sidebar header
    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        ">
            <h2 style="margin: 0; font-size: 24px;">⚙️ CONTROL PANEL</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Mode selection with better styling
    st.sidebar.markdown("### 📡 Operation Mode")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Live Camera", "Upload Conveyor Video", "Streamlit Camera (Click-to-Capture)"],
        help="Live Camera = Real-time streaming | Streamlit Camera = Click to capture each frame",
        label_visibility="collapsed"
    )
    
    if mode == "Streamlit Camera (Click-to-Capture)":
        st.sidebar.warning("⚠️ **Note**: Streamlit Camera requires clicking to capture each frame. For true real-time streaming, use 'Live Camera' mode.")
    
    st.sidebar.markdown("---")
    
    # Mode-specific settings
    if mode == "Live Camera":
        camera_index = st.sidebar.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=0,
            help="Camera device index (usually 0 for default camera)"
        )
        st.sidebar.markdown("---")
        roi, trigger_x = get_roi_settings()
        
        return {
            'mode': mode,
            'camera_index': camera_index,
            'roi': roi,
            'trigger_line_x': trigger_x
        }
    
    elif mode == "Streamlit Camera (Click-to-Capture)":
        st.sidebar.info("💡 Streamlit Camera uses browser permissions, which are often easier to grant than system permissions.")
        st.sidebar.markdown("---")
        roi, trigger_x = get_roi_settings()
        
        return {
            'mode': mode,
            'roi': roi,
            'trigger_line_x': trigger_x
        }
    
    else:  # Upload Video mode
        uploaded_file = st.sidebar.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process"
        )
        st.sidebar.markdown("---")
        roi, trigger_x = get_roi_settings()
        
        return {
            'mode': mode,
            'uploaded_file': uploaded_file,
            'roi': roi,
            'trigger_line_x': trigger_x
        }


def display_metrics(label: Optional[str], confidence: float) -> None:
    """Display real-time metrics with large, clear indicators for operators."""
    # Main status card - large and prominent
    if label:
        category_type, color, message = get_segregation_info(label)
        text_color = '#000000' if category_type == 'recyclable' else '#FFFFFF'
        
        # Large status card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                border: 3px solid {color};
            ">
                <h1 style="color: {text_color}; font-size: 48px; margin: 10px 0; font-weight: bold;">
                    {message}
                </h1>
                <p style="color: {text_color}; font-size: 24px; margin: 5px 0; font-weight: 600;">
                    {label.upper().replace('_', ' ')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Confidence indicator
        confidence_color = "#00FF00" if confidence >= 0.8 else "#FFA500" if confidence >= 0.5 else "#FF6B6B"
        st.markdown(
            f"""
            <div style="
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            ">
                <h3 style="margin: 0 0 10px 0; color: #333;">Confidence Level</h3>
                <div style="
                    background-color: #e0e0e0;
                    border-radius: 10px;
                    height: 40px;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        background: linear-gradient(90deg, {confidence_color} 0%, {confidence_color}dd 100%);
                        width: {confidence * 100}%;
                        height: 100%;
                        border-radius: 10px;
                        transition: width 0.3s ease;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        font-size: 18px;
                    ">
                        {confidence:.0%}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Waiting state
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #808080 0%, #666666 100%);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 20px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            ">
                <h1 style="color: white; font-size: 48px; margin: 10px 0; font-weight: bold;">
                    ⏳ WAITING
                </h1>
                <p style="color: white; font-size: 20px; margin: 5px 0;">
                    Place item on conveyor belt
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style="
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            ">
                <h3 style="margin: 0 0 10px 0; color: #333;">Confidence Level</h3>
                <div style="
                    background-color: #e0e0e0;
                    border-radius: 10px;
                    height: 40px;
                ">
                    <div style="
                        background-color: #ccc;
                        width: 0%;
                        height: 100%;
                        border-radius: 10px;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_segregation_guide(label: Optional[str]) -> None:
    """Display segregation guide with clear bin assignment for operators."""
    st.markdown("### 🗑️ BIN ASSIGNMENT")
    
    if label:
        category_type, color, message = get_segregation_info(label)
        text_color = '#000000' if category_type == 'recyclable' else '#FFFFFF'
        
        # Bin assignment card
        bin_name = "RECYCLABLES BIN" if category_type == "recyclable" else "GENERAL WASTE BIN"
        bin_icon = "♻️" if category_type == "recyclable" else "🗑️"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                margin: 15px 0;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                border: 3px solid {color};
            ">
                <div style="font-size: 48px; margin-bottom: 10px;">
                    {bin_icon}
                </div>
                <h2 style="color: {text_color}; font-size: 28px; margin: 10px 0; font-weight: bold;">
                    {bin_name}
                </h2>
                <p style="color: {text_color}; font-size: 18px; margin: 5px 0; opacity: 0.9;">
                    Place item here
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Quick reference
        if category_type == "recyclable":
            st.success("""
            **✅ RECYCLABLES BIN**
            - Plastic containers
            - Metal cans
            - Glass bottles
            """)
        elif category_type == "general_waste":
            st.warning("""
            **⚠️ GENERAL WASTE BIN**
            - Cardboard
            - Paper
            - Organic waste
            - Other non-recyclables
            """)
    else:
        # Default state - show both bins
        st.markdown(
            """
            <div style="
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                margin: 15px 0;
            ">
                <p style="color: #666; font-size: 18px; margin: 0;">
                    Waiting for item classification...
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show both bins as reference
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #00FF00 0%, #00CC00 100%);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    border: 2px solid #00FF00;
                ">
                    <div style="font-size: 32px; margin-bottom: 5px;">♻️</div>
                    <p style="color: #000; font-size: 14px; font-weight: bold; margin: 0;">
                        RECYCLABLES
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #FFA500 0%, #FF8800 100%);
                    padding: 15px;
                    border-radius: 10px;
                    text-align: center;
                    border: 2px solid #FFA500;
                ">
                    <div style="font-size: 32px; margin-bottom: 5px;">🗑️</div>
                    <p style="color: #000; font-size: 14px; font-weight: bold; margin: 0;">
                        GENERAL WASTE
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


def open_camera(camera_index: int) -> Optional[cv2.VideoCapture]:
    """
    Open camera with multiple backend attempts for macOS compatibility.
    
    Args:
        camera_index: Camera device index
        
    Returns:
        VideoCapture object if successful, None otherwise
    """
    for backend_id, backend_name in CAMERA_BACKENDS:
        try:
            cap = cv2.VideoCapture(camera_index, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if not st.session_state.camera_opened_message_shown:
                        st.success(f"✅ Camera {camera_index} opened using {backend_name}")
                        st.session_state.camera_opened_message_shown = True
                    return cap
                else:
                    cap.release()
        except Exception:
            if cap:
                cap.release()
            continue
    
    return None


def show_camera_error(camera_index: int) -> None:
    """Display camera error message with troubleshooting steps."""
    st.error(f"❌ Could not open camera {camera_index}")
    st.error("**Troubleshooting Steps:**")
    st.error("1. **Check Camera Permissions**:")
    st.error("   - System Settings → Privacy & Security → Camera")
    st.error("   - Enable Terminal/IDE/Python")
    st.error("   - Restart Streamlit after granting permissions")
    st.error("")
    st.error("2. **Close Other Apps**: Make sure no other app is using the camera")
    st.error("   (FaceTime, Photo Booth, Zoom, etc.)")
    st.error("")
    st.error("3. **Try Different Camera Index**:")
    st.error("   - Try 1, 2, 3 instead of 0")
    st.error("   - Check sidebar for camera index selector")
    st.error("")
    st.error("4. **Try Streamlit Camera Mode**: Uses browser permissions (easier)")
    st.error("")
    st.error("5. **Test Camera Access**:")
    st.error("   ```bash")
    st.error("   python utils/test_camera.py")
    st.error("   ```")


def update_classification_metrics(frame: Optional[np.ndarray] = None) -> None:
    """
    Update session state with classification results from tracked objects.
    Also captures images for objects with good confidence.
    
    Args:
        frame: Current frame to capture object images from (optional)
    """
    if st.session_state.engine is None:
        return
    
    for obj_id, tracked_data in st.session_state.engine.tracked_objects.items():
        if tracked_data.get('label') and tracked_data.get('has_crossed'):
            label = tracked_data['label']
            confidence = tracked_data.get('confidence', 0.0)
            
            # Update current metrics
            st.session_state.current_label = label
            st.session_state.current_confidence = confidence
            
            # Capture image if confidence is above threshold and frame is available
            # Allow multiple detections of same object type (don't check for duplicates)
            if (confidence >= st.session_state.min_confidence_threshold and 
                frame is not None and 
                obj_id in st.session_state.engine.tracked_objects):
                
                # Check if we recently captured this specific object (within 1 second) to prevent spam
                recent_capture = any(
                    item.get('object_id') == obj_id and 
                    (time.time() - item.get('timestamp', 0)) < 1.0
                    for item in st.session_state.classification_history
                )
                
                if not recent_capture:
                    # Extract object region from frame
                    try:
                        object_image = extract_object_image(frame, obj_id, tracked_data)
                        
                        if object_image is not None:
                            # Convert to RGB for display
                            if len(object_image.shape) == 3 and object_image.shape[2] == 3:
                                object_image_rgb = cv2.cvtColor(object_image, cv2.COLOR_BGR2RGB)
                            else:
                                object_image_rgb = object_image
                            
                            # Add to history (keep last 50 items for side-by-side display)
                            history_item = {
                                'timestamp': time.time(),
                                'object_id': obj_id,
                                'label': label,
                                'confidence': confidence,
                                'image': object_image_rgb,
                            }
                            st.session_state.classification_history.insert(0, history_item)
                            
                            # Update session statistics - ONLY if this object hasn't been counted before
                            if obj_id not in st.session_state.session_stats['counted_object_ids']:
                                st.session_state.session_stats['total_items'] += 1
                                category_type, _, _ = get_segregation_info(label)
                                if category_type == 'recyclable':
                                    st.session_state.session_stats['recyclables_count'] += 1
                                elif category_type == 'general_waste':
                                    st.session_state.session_stats['general_waste_count'] += 1
                                st.session_state.session_stats['items_by_type'][label] += 1
                                # Mark this object ID as counted
                                st.session_state.session_stats['counted_object_ids'].add(obj_id)
                            
                            # Keep only last 50 items (increased for side-by-side display)
                            if len(st.session_state.classification_history) > 50:
                                st.session_state.classification_history = st.session_state.classification_history[:50]
                    except Exception as e:
                        # Silently fail if extraction fails
                        pass
            
            break


def extract_object_image(frame: np.ndarray, obj_id: int, tracked_data: Dict) -> Optional[np.ndarray]:
    """
    Extract object image from frame based on tracked object data.
    
    Args:
        frame: Full frame image
        obj_id: Object ID
        tracked_data: Tracked object data dictionary
        
    Returns:
        Extracted object image or None
    """
    try:
        # Try to find the object in the engine's current detected objects to get bbox
        bbox = None
        if st.session_state.engine is not None:
            # The engine processes objects and we need to get the bbox
            # For now, we'll use centroid-based extraction with a reasonable size
            # In a future improvement, we could store bbox in tracked_data
            pass
        
        # Extract region around centroid
        if 'centroid' in tracked_data:
            cx, cy = tracked_data['centroid']
            
            # Extract a square region around centroid (larger for better view)
            # Size based on typical object size on conveyor belt
            size = 250  # Size of extracted region (increased for better visibility)
            h, w = frame.shape[:2]
            
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)
            
            if x2 > x1 and y2 > y1:
                object_roi = frame[y1:y2, x1:x2].copy()
                
                # Resize if too small for better display
                if object_roi.shape[0] < 100 or object_roi.shape[1] < 100:
                    scale = max(100 / object_roi.shape[0], 100 / object_roi.shape[1])
                    new_w = int(object_roi.shape[1] * scale)
                    new_h = int(object_roi.shape[0] * scale)
                    object_roi = cv2.resize(object_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Draw a border to highlight the object
                cv2.rectangle(object_roi, (0, 0), (object_roi.shape[1]-1, object_roi.shape[0]-1), (0, 255, 0), 3)
                return object_roi
        
        return None
    except Exception:
        return None


def display_classification_history() -> None:
    """Display classification history in the sidebar."""
    # Only show if we have history or are in live camera mode
    if not st.session_state.classification_history:
        # Show empty state only in live camera mode
        if st.session_state.processing_active:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📸 Detected Objects")
            st.sidebar.info("No objects detected yet. Objects will appear here when detected.")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📸 Detected Objects")
    
    # Confidence threshold slider (use unique key to prevent widget conflicts)
    min_conf = st.sidebar.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.min_confidence_threshold,
        step=0.05,
        key="min_confidence_slider",
        help="Minimum confidence to show in detected objects list"
    )
    st.session_state.min_confidence_threshold = min_conf
    
    # Filter by confidence
    filtered_history = [
        item for item in st.session_state.classification_history
        if item['confidence'] >= st.session_state.min_confidence_threshold
    ]
    
    if not filtered_history:
        st.sidebar.info("No objects detected above confidence threshold")
        return
    
    # Show count and grouping option
    st.sidebar.caption(f"Found {len(filtered_history)} object(s) with confidence ≥ {st.session_state.min_confidence_threshold:.0%}")
    
    # Group by label option
    group_by_type = st.sidebar.checkbox("Group by Type", value=True, key="group_by_type")
    
    if group_by_type:
        # Group objects by label type
        grouped = defaultdict(list)
        for item in filtered_history:
            grouped[item['label']].append(item)
        
        # Display grouped objects side by side
        for label, items in grouped.items():
            st.sidebar.markdown(f"### {label.upper()} ({len(items)})")
            
            # Show items in a grid (2 columns)
            num_items = len(items)
            for i in range(0, num_items, 2):
                cols = st.sidebar.columns(2)
                for j, col in enumerate(cols):
                    if i + j < num_items:
                        item = items[i + j]
                        with col:
                            # Display image
                            st.image(
                                item['image'], 
                                use_container_width=True,
                                caption=f"{item['confidence']:.0%}"
                            )
                            
                            # Show label and confidence
                            category_type, color, message = get_segregation_info(item['label'])
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: {color};
                                    padding: 5px;
                                    border-radius: 3px;
                                    text-align: center;
                                    color: {'#000000' if category_type == 'recyclable' else '#FFFFFF'};
                                    font-size: 10px;
                                    margin: 2px 0;
                                ">
                                    {item['label'].upper()}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Timestamp
                            time_ago = time.time() - item['timestamp']
                            if time_ago < 60:
                                st.caption(f"{int(time_ago)}s ago")
                            else:
                                st.caption(f"{int(time_ago/60)}m ago")
    else:
        # Display all objects side by side in grid (2 columns)
        num_items = len(filtered_history)
        for i in range(0, num_items, 2):
            cols = st.sidebar.columns(2)
            for j, col in enumerate(cols):
                if i + j < num_items:
                    item = filtered_history[i + j]
                    with col:
                        # Display image
                        st.image(
                            item['image'], 
                            use_container_width=True,
                            caption=f"#{i+j+1}: {item['label'].upper()} ({item['confidence']:.0%})"
                        )
                        
                        # Show segregation info
                        category_type, color, message = get_segregation_info(item['label'])
                        st.markdown(
                            f"""
                            <div style="
                                background-color: {color};
                                padding: 5px;
                                border-radius: 3px;
                                text-align: center;
                                color: {'#000000' if category_type == 'recyclable' else '#FFFFFF'};
                                font-size: 10px;
                                margin: 2px 0;
                            ">
                                {message}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Timestamp
                        time_ago = time.time() - item['timestamp']
                        if time_ago < 60:
                            st.caption(f"{int(time_ago)}s ago")
                        else:
                            st.caption(f"{int(time_ago/60)}m ago")
    
    # Clear history button
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear History", key="clear_history", use_container_width=True):
        st.session_state.classification_history = []
        st.rerun()


def process_live_camera(camera_index: int, roi: Optional[Tuple], trigger_line_x: Optional[int]) -> None:
    """Process live camera feed in real-time with stable streaming."""
    # Initialize components if needed
    if st.session_state.classifier is None:
        st.session_state.classifier = load_classifier()
    
    if st.session_state.engine is None:
        st.session_state.engine = ConveyorEngine(
            classifier=st.session_state.classifier,
            roi=roi,
            trigger_line_x=trigger_line_x,
            classification_cooldown=DEFAULT_CLASSIFICATION_COOLDOWN
        )
    else:
        # Update engine settings
        if roi:
            st.session_state.engine.roi = roi
        if trigger_line_x:
            st.session_state.engine.trigger_line_x = trigger_line_x
    
    # Open camera
    if st.session_state.camera is None or not st.session_state.camera.isOpened():
        st.session_state.camera = open_camera(camera_index)
        
        if st.session_state.camera is None:
            show_camera_error(camera_index)
            st.session_state.processing_active = False
            return
    
    # Create placeholder for video (persistent across reruns)
    video_placeholder = st.empty()
    
    # Frame rate control
    fps = st.sidebar.slider("Frame Rate (FPS)", MIN_FPS, MAX_FPS, DEFAULT_FPS, 1, key="camera_fps")
    frame_delay = 1.0 / fps if fps > 0 else 0.1
    
    # Motion detection settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎬 Motion Detection")
    
    if st.session_state.engine is not None:
        # Motion threshold (sensitivity)
        motion_threshold = st.sidebar.slider(
            "Motion Sensitivity",
            min_value=0.01,
            max_value=0.10,
            value=st.session_state.engine.motion_threshold,
            step=0.01,
            key="motion_threshold",
            help="Higher = less sensitive (requires more change to detect motion)"
        )
        st.session_state.engine.motion_threshold = motion_threshold
        
        # Motion status
        motion_status = "✅ Motion Detected" if st.session_state.engine.motion_detected else "⏸️ No Motion"
        motion_color = "🟢" if st.session_state.engine.motion_detected else "🔴"
        st.sidebar.markdown(f"{motion_color} **{motion_status}**")
        st.sidebar.caption("⚠️ Classification only occurs when motion is detected")
    else:
        st.sidebar.info("Start camera to see motion detection status")
    
    # Initialize frame tracking
    if st.session_state.camera_last_update == 0:
        st.session_state.camera_last_update = time.time()
    
    # Use longer rerun interval to reduce flickering
    rerun_interval = st.sidebar.slider(
        "Update Interval (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=RERUN_INTERVAL,
        step=0.5,
        key="rerun_interval",
        help="Higher = less flicker but less responsive UI updates"
    )
    
    start_time = time.time()
    last_display_update = time.time()
    # Display update interval - update display at max 10 FPS to reduce flicker
    display_update_interval = max(0.1, frame_delay * 2)  # At least 100ms or 2x frame delay
    
    try:
        # Process frames continuously for the rerun interval
        frames_processed = 0
        while (time.time() - start_time) < rerun_interval and st.session_state.processing_active:
            ret, frame = st.session_state.camera.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            # Process frame
            processed_frame = st.session_state.engine.process_frame(frame)
            
            # Update metrics and capture images
            update_classification_metrics(frame)
            
            # Convert BGR to RGB for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Store latest frame for display
            st.session_state.last_display_frame = processed_frame_rgb
            
            # Update video display at controlled rate (reduces flicker)
            current_time = time.time()
            if current_time - last_display_update >= display_update_interval:
                # Always show the latest frame
                if st.session_state.last_display_frame is not None:
                    video_placeholder.image(
                        st.session_state.last_display_frame, 
                        channels="RGB", 
                        use_container_width=True
                    )
                last_display_update = current_time
                st.session_state.camera_last_update = current_time
            
            frames_processed += 1
            st.session_state.camera_frame_count += 1
            
            # Show FPS in sidebar (update every 30 frames)
            if st.session_state.camera_frame_count % 30 == 0:
                elapsed = time.time() - (current_time - display_update_interval * 30)
                actual_fps = 30 / elapsed if elapsed > 0 else fps
                st.sidebar.metric("Actual FPS", f"{actual_fps:.1f}")
            
            # Sleep to maintain target frame rate (but don't sleep too long)
            sleep_time = min(frame_delay, 0.03)  # Cap sleep at 30ms to maintain responsiveness
            time.sleep(sleep_time)
        
        # Ensure last frame is displayed before rerun
        if st.session_state.last_display_frame is not None:
            video_placeholder.image(
                st.session_state.last_display_frame,
                channels="RGB",
                use_container_width=True
            )
        
        # Display classification history in sidebar (after processing loop, once per rerun)
        display_classification_history()
        
        # Continue processing if still active (with minimal delay before rerun)
        if st.session_state.processing_active:
            # Small delay to prevent excessive reruns
            time.sleep(0.05)
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error processing camera feed: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.session_state.processing_active = False
    finally:
        # Only release camera if explicitly stopped
        if not st.session_state.processing_active and st.session_state.camera:
            try:
                st.session_state.camera.release()
            except Exception:
                pass
            st.session_state.camera = None


def process_streamlit_camera(roi: Optional[Tuple], trigger_line_x: Optional[int]) -> None:
    """
    Process frames from Streamlit's built-in camera widget.
    
    NOTE: st.camera_input is designed for single photo capture, not continuous streaming.
    For true real-time video, use 'Live Camera' mode which uses OpenCV VideoCapture.
    """
    # Initialize components if needed
    if st.session_state.classifier is None:
        st.session_state.classifier = load_classifier()
    
    if st.session_state.engine is None:
        st.session_state.engine = ConveyorEngine(
            classifier=st.session_state.classifier,
            roi=roi,
            trigger_line_x=trigger_line_x,
            classification_cooldown=DEFAULT_CLASSIFICATION_COOLDOWN
        )
    else:
        # Update engine settings
        if roi:
            st.session_state.engine.roi = roi
        if trigger_line_x:
            st.session_state.engine.trigger_line_x = trigger_line_x
    
    # Important notice about limitations
    st.info("📝 **How it works**: Click the camera button below to capture a frame. Each click processes one frame.")
    st.warning("⚠️ **Limitation**: Streamlit's camera widget cannot stream continuously. For real-time video, use 'Live Camera' mode.")
    
    # Create placeholders for display
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Use Streamlit's camera input (click-to-capture)
    camera_image = st.camera_input("📷 Click to capture frame", key="waste_camera_stable")
    
    if camera_image is not None:
        try:
            # Convert PIL Image to numpy array properly
            if hasattr(camera_image, 'read'):
                camera_image = Image.open(camera_image)
            
            # Convert PIL Image to numpy array
            frame = np.array(camera_image)
            
            # Ensure it's a valid numpy array with correct shape
            if frame is None or len(frame.shape) < 2:
                st.error("❌ Failed to convert camera image to numpy array")
                return
            
            # Handle different image formats
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame
            processed_frame = st.session_state.engine.process_frame(frame_bgr)
            
            # Update metrics and capture images
            update_classification_metrics(frame_bgr)
            
            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display processed frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            # Show classification result if available
            if st.session_state.current_label:
                status_placeholder.success(
                    f"✅ Classified: **{st.session_state.current_label}** "
                    f"(confidence: {st.session_state.current_confidence:.1%})"
                )
            else:
                status_placeholder.info("👀 Processing... Place items on the conveyor belt.")
            
            # Show instructions for next capture
            st.info("💡 **Click the camera button again** to capture and process the next frame.")
            
            # Display classification history in sidebar
            display_classification_history()
            
        except Exception as e:
            st.error(f"❌ Error processing camera image: {e}")
            st.error("Please try capturing again or use 'Live Camera' mode for continuous streaming.")
    else:
        st.info("👆 **Click the camera button above** to capture and classify waste items.")
        st.info("💡 The camera will ask for permission - click 'Allow' to proceed.")
        st.info("🔄 **For continuous real-time feed**: Switch to 'Live Camera' mode in the sidebar.")


def process_uploaded_video(uploaded_file, roi: Optional[Tuple], trigger_line_x: Optional[int]) -> None:
    """Process uploaded video file."""
    # Initialize components if needed
    if st.session_state.classifier is None:
        st.session_state.classifier = load_classifier()
    
    if st.session_state.engine is None:
        st.session_state.engine = ConveyorEngine(
            classifier=st.session_state.classifier,
            roi=roi,
            trigger_line_x=trigger_line_x,
            classification_cooldown=DEFAULT_CLASSIFICATION_COOLDOWN
        )
    else:
        # Update engine settings
        if roi:
            st.session_state.engine.roi = roi
        if trigger_line_x:
            st.session_state.engine.trigger_line_x = trigger_line_x
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Create placeholder for video
        video_placeholder = st.empty()
        
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        last_update_time = time.time()
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = st.session_state.engine.process_frame(frame)
            
            # Update metrics and capture images
            update_classification_metrics(frame)
            
            # Convert BGR to RGB for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update video display
            current_time = time.time()
            if current_time - last_update_time >= frame_delay:
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
                last_update_time = current_time
            
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress:.1%})")
        
        cap.release()
        progress_bar.empty()
        status_text.success("✅ Video processing complete!")
        
    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def reset_camera_state() -> None:
    """Reset camera-related session state."""
    st.session_state.current_label = None
    st.session_state.current_confidence = 0.0
    st.session_state.engine = None
    st.session_state.camera_opened_message_shown = False
    st.session_state.camera_last_update = 0
    st.session_state.camera_frame_count = 0


def display_session_statistics() -> None:
    """Display session statistics dashboard for operators."""
    stats = st.session_state.session_stats
    
    st.markdown("### 📊 Session Statistics")
    
    # Main stats in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            ">
                <h2 style="margin: 0; font-size: 36px; font-weight: bold;">
                    {stats['total_items']}
                </h2>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">
                    Total Items
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #00FF00 0%, #00CC00 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: #000;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            ">
                <h2 style="margin: 0; font-size: 36px; font-weight: bold;">
                    {stats['recyclables_count']}
                </h2>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">
                    ♻️ Recyclables
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFA500 0%, #FF8800 100%);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                color: #000;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            ">
                <h2 style="margin: 0; font-size: 36px; font-weight: bold;">
                    {stats['general_waste_count']}
                </h2>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">
                    🗑️ General Waste
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Items by type breakdown
    if stats['items_by_type']:
        st.markdown("#### Breakdown by Type")
        type_cols = st.columns(min(3, len(stats['items_by_type'])))
        for idx, (item_type, count) in enumerate(list(stats['items_by_type'].items())[:6]):
            with type_cols[idx % 3]:
                st.metric(
                    label=item_type.upper().replace('_', ' '),
                    value=count
                )


def main() -> None:
    """Main application function."""
    # Professional header with custom styling
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .main-header h1 {
            margin: 0;
            font-size: 48px;
            font-weight: bold;
        }
        .main-header p {
            margin: 10px 0 0 0;
            font-size: 20px;
            opacity: 0.9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="main-header">
            <h1>♻️ Waste Segregation System</h1>
            <p>AI-Powered Automated Sorting</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    settings = sidebar()
    
    # Main content area - optimized layout for operators
    col_video, col_info = st.columns([3, 2])
    
    with col_video:
        st.markdown("### 📹 Live Conveyor Feed")
        
        if settings['mode'] == "Live Camera":
            # Professional status banner
            if st.session_state.processing_active:
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
                        padding: 15px;
                        border-radius: 10px;
                        color: white;
                        text-align: center;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    ">
                        <h3 style="margin: 0; font-size: 20px;">🟢 SYSTEM ACTIVE - Processing Live Feed</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(135deg, #FF4444 0%, #CC0000 100%);
                        padding: 15px;
                        border-radius: 10px;
                        color: white;
                        text-align: center;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    ">
                        <h3 style="margin: 0; font-size: 20px;">⏸️ SYSTEM STANDBY - Click Start to Begin</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Large, prominent control buttons
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("▶️ START SYSTEM", type="primary", use_container_width=True):
                    st.session_state.processing_active = True
                    reset_camera_state()
                    st.rerun()
            
            with col_stop:
                if st.button("⏹️ STOP SYSTEM", use_container_width=True):
                    st.session_state.processing_active = False
                    st.session_state.camera_opened_message_shown = False
                    if st.session_state.camera:
                        try:
                            st.session_state.camera.release()
                        except Exception:
                            pass
                        st.session_state.camera = None
                    st.rerun()
            
            # Process camera if active
            if st.session_state.processing_active:
                process_live_camera(
                    settings['camera_index'],
                    settings['roi'],
                    settings['trigger_line_x']
                )
            else:
                st.info("👆 Click '▶️ Start Camera' to begin **real-time streaming**")
                st.info("🔄 **This mode streams continuously** - no clicking needed after starting!")
                
                # Quick setup guide
                with st.expander("🚀 Quick Setup Guide", expanded=True):
                    st.markdown("""
                    **Step 1: Grant Camera Permissions**
                    1. Open **System Settings** → **Privacy & Security** → **Camera**
                    2. Enable **Terminal** (or your IDE)
                    3. **Restart Streamlit** after granting permissions
                    
                    **Step 2: Test Camera**
                    ```bash
                    python utils/test_camera.py
                    ```
                    
                    **Step 3: Start Streaming**
                    - Click "▶️ Start Camera" above
                    - Feed will stream automatically!
                    """)
                
                with st.expander("🔧 Camera Not Working? Troubleshooting"):
                    st.markdown("""
                    **Quick Fixes:**
                    1. **Grant Camera Permissions**: 
                       - System Settings → Privacy & Security → Camera
                       - Enable Terminal/IDE/Python
                       - **Restart Streamlit** after granting
                    
                    2. **Test Your Camera**:
                       ```bash
                       python utils/test_camera.py
                       ```
                       Or run: `bash utils/fix_camera_permissions.sh`
                    
                    3. **Try Different Camera Index**: 
                       - Change from 0 to 1, 2, or 3 in sidebar
                    
                    4. **Close Other Apps**: 
                       - FaceTime, Photo Booth, Zoom, etc.
                    
                    5. **Check Camera Hardware**:
                       - Try Photo Booth app to verify camera works
                       - If Photo Booth doesn't work, it's a system issue
                    
                    **See `LIVE_CAMERA_SETUP.md` for detailed step-by-step guide.**
                    """)
                st.warning("💡 **Tip**: If camera access fails, try 'Streamlit Camera' mode which uses browser permissions instead of system permissions.")
        
        elif settings['mode'] == "Streamlit Camera (Click-to-Capture)":
            st.warning("⚠️ **Limitation**: Streamlit Camera requires clicking to capture each frame. It cannot stream continuously.")
            st.info("💡 **For Real-time Streaming**: Use 'Live Camera' mode instead (requires system camera permissions).")
            process_streamlit_camera(
                settings['roi'],
                settings['trigger_line_x']
            )
        
        else:  # Upload Video mode
            if settings['uploaded_file'] is not None:
                st.session_state.processing_active = True
                reset_camera_state()
                
                process_uploaded_video(
                    settings['uploaded_file'],
                    settings['roi'],
                    settings['trigger_line_x']
                )
            else:
                st.info("👆 Upload a video file from the sidebar to begin processing")
    
    with col_info:
        # Current item status - large and clear
        st.markdown("### 🎯 CURRENT ITEM STATUS")
        display_metrics(
            st.session_state.current_label,
            st.session_state.current_confidence
        )
        
        st.markdown("---")
        
        # Bin assignment - clear and prominent
        display_segregation_guide(st.session_state.current_label)
        
        st.markdown("---")
        
        # Session statistics
        display_session_statistics()
        
        # Reset statistics button
        if st.button("🔄 Reset Statistics", key="reset_stats", use_container_width=True):
            st.session_state.session_stats = {
                'total_items': 0,
                'recyclables_count': 0,
                'general_waste_count': 0,
                'items_by_type': defaultdict(int),
                'counted_object_ids': set(),
            }
            st.rerun()
    
    # Footer - minimal and professional
    st.markdown("---")
    st.markdown(
        f"""
        <div style='
            text-align: center; 
            color: #999; 
            padding: 15px;
            font-size: 12px;
        '>
            <p style="margin: 5px 0;">Powered by PyTorch MPS • Mac Mini M4 Optimized</p>
            <p style="margin: 5px 0;">Model: {MODEL_NAME.split('/')[-1]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
