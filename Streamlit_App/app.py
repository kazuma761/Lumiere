import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import re
import time
import os

# Ensure the directory exists
capture_directory = "Captured_Images"
if not os.path.exists(capture_directory):
    os.makedirs(capture_directory)

st.set_page_config(
    page_title="Lumiere",
    page_icon="NYRA Logo.png",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Facial Landmarks Definitions
LEFT_CHEEK_CONTOUR = [116, 123, 147, 192, 176, 149, 150, 136, 172, 58, 132]  # Higher on left cheek
RIGHT_CHEEK_CONTOUR = [345, 352, 376, 421, 405, 378, 379, 365, 397, 288, 361]  # Higher on right cheek
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LEFT_EYESHADOW = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
RIGHT_EYESHADOW = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Facial landmark indices
SKIN_LANDMARK = 10
LEFT_IRIS = [474, 475, 476, 477]  # Iris contour points
RIGHT_IRIS = [469, 470, 471, 472]  # Iris contour points
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# UI Styling
st.markdown(
    """
    <style>
    /* Main title styling */
    .main-title {
        color: #2E4057;
        font-size: 3em;
        font-weight: 700;
        padding: 20px 0;
        text-align: center;
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 2rem 1rem;
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        padding: 10px 15px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2em;
        color: #2E4057;
        font-weight: 600;
        margin: 20px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    hr {
        border: 1px solid #eee;
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown('<h1 class="main-title">Lumiere - Touchup Assistant</h1>', unsafe_allow_html=True)

# Sidebar Header Styling
st.sidebar.markdown('<div class="section-header">Camera Controls</div>', unsafe_allow_html=True)

# Streamlit UI
st.title("NYRA - Virtual AI Makeup Assistant")

# Webcam Controls
st.sidebar.title("📷 Webcam Controls")
if "webcam_enabled" not in st.session_state:
    st.session_state.webcam_enabled = False

if st.sidebar.button("🎥 Start Webcam" if not st.session_state.webcam_enabled else "🛑 Stop Webcam"):
    st.session_state.webcam_enabled = not st.session_state.webcam_enabled

if st.session_state.webcam_enabled:
    st.sidebar.success("Webcam is ON")
else:
    st.sidebar.error("Webcam is OFF")

# Capture Image Button
if st.sidebar.button("📸 Capture & Save Image"):
    st.session_state.capture_image = True

if "capture_image" not in st.session_state:
    st.session_state.capture_image = False

# Utility Functions
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def bgr_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])

# Makeup Controls
st.sidebar.title("Touch up Controls")

# Blush Settings
st.sidebar.markdown('<div class="section-header">✨ Blush</div>', unsafe_allow_html=True)
if "blush_enabled" not in st.session_state:
    st.session_state.blush_enabled = False
if st.sidebar.button("Enable Blush" if not st.session_state.blush_enabled else "❌ Disable Blush"):
    st.session_state.blush_enabled = not st.session_state.blush_enabled
blush_opacity = st.sidebar.slider("Blush Intensity", 0.1, 1.0, 0.2)
blush_color = st.sidebar.color_picker("Blush Color", "#BB704C")

# Lipstick Settings
st.sidebar.markdown('<div class="section-header">✨ Lips</div>', unsafe_allow_html=True)
if "lipstick_enabled" not in st.session_state:
    st.session_state.lipstick_enabled = False
if st.sidebar.button("Enable Lipstick" if not st.session_state.lipstick_enabled else "❌ Disable Lipstick"):
    st.session_state.lipstick_enabled = not st.session_state.lipstick_enabled
lipstick_opacity = st.sidebar.slider("Lipstick Intensity", 0.1, 0.5, 0.13)
lipstick_color = st.sidebar.color_picker("Lipstick Color", "#A44D4D")

# Eye Settings
st.sidebar.markdown('<div class="section-header">✨ Eyes</div>', unsafe_allow_html=True)

# Iris Color Controls
if "iris_enabled" not in st.session_state:
    st.session_state.iris_enabled = False
if st.sidebar.button("Enable Iris Color" if not st.session_state.iris_enabled else "❌ Disable Iris Color"):
    st.session_state.iris_enabled = not st.session_state.iris_enabled
iris_color = st.sidebar.color_picker("Iris Color", "#060D23")
blur_intensity = st.sidebar.slider("Iris Blur", 1, 25, 21, step=2)

# Eyeshadow Controls
if "eyeshadow_enabled" not in st.session_state:
    st.session_state.eyeshadow_enabled = False
if st.sidebar.button("Enable Eyeshadow" if not st.session_state.eyeshadow_enabled else "❌ Disable Eyeshadow"):
    st.session_state.eyeshadow_enabled = not st.session_state.eyeshadow_enabled
eyeshadow_color = st.sidebar.color_picker("Eyeshadow Color", "#7A4B3A")
eyeshadow_opacity = st.sidebar.slider("Eyeshadow Intensity", 0.1, 1.0, 0.14)

# Face Enhancement Controls
st.sidebar.markdown('<div class="section-header">✨ Skin</div>', unsafe_allow_html=True)
smoothing_intensity = st.sidebar.slider("Smooth Skin Level", 0, 100, 15)
brightness_increase = st.sidebar.slider("Increase Brightness", 0, 100, 4)

# Create a better layout for the main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Main display window
    FRAME_WINDOW = st.image([], use_column_width=True)

with col2:
    st.markdown('<div class="section-header">Status</div>', unsafe_allow_html=True)
    if st.session_state.webcam_enabled:
        st.success("✨ Camera Active")
    else:
        st.error("📷 Camera Inactive")
        
    # Add some helpful tips
    st.markdown('<div class="section-header">Tips</div>', unsafe_allow_html=True)
    st.info("""
    💡 **Quick Tips:**
    - Enable effects from the sidebar
    - Adjust intensities using sliders
    - Capture images anytime
    - Face should be well-lit
    """)

if st.session_state.webcam_enabled:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access the webcam.")
            break

        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                # Create face mask
                face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                face_points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] 
                                      for i in FACE_OVAL], np.int32)
                cv2.fillPoly(face_mask, [face_points], 255)                # Apply Blush
                if st.session_state.blush_enabled:
                    # Get face width for scaling
                    face_width = abs(int(face_landmarks.landmark[454].x * w) - int(face_landmarks.landmark[234].x * w))
                    
                    # Calculate blush centers (higher on the cheeks)
                    left_blush_center = np.array([
                        int(face_landmarks.landmark[116].x * w),  # X coordinate
                        int((face_landmarks.landmark[123].y + face_landmarks.landmark[147].y) * h / 2)  # Y coordinate
                    ])
                    
                    right_blush_center = np.array([
                        int(face_landmarks.landmark[345].x * w),  # X coordinate
                        int((face_landmarks.landmark[352].y + face_landmarks.landmark[376].y) * h / 2)  # Y coordinate
                    ])
                    
                    # Create blush overlay
                    blush_overlay = np.zeros_like(frame)
                    
                    # Calculate radius (smaller and more focused)
                    radius = int(face_width * 0.15)  # Adjust this value to change blush size
                    
                    # Draw circular blush
                    cv2.circle(blush_overlay, tuple(left_blush_center), radius, hex_to_bgr(blush_color), -1)
                    cv2.circle(blush_overlay, tuple(right_blush_center), radius, hex_to_bgr(blush_color), -1)
                    
                    # Apply stronger Gaussian blur for softer edges
                    blush_overlay = cv2.GaussianBlur(blush_overlay, (55, 55), 20)
                    
                    # Create and blur the mask
                    blush_mask = np.zeros_like(frame[:,:,0])
                    cv2.circle(blush_mask, tuple(left_blush_center), radius, 255, -1)
                    cv2.circle(blush_mask, tuple(right_blush_center), radius, 255, -1)
                    blush_mask = cv2.GaussianBlur(blush_mask, (55, 55), 20)
                    
                    # Apply blush with adjusted opacity
                    alpha = (blush_mask / 255.0) * blush_opacity * 0.8  # Adjusted opacity for more natural look
                    for c in range(3):
                        frame[:,:,c] = frame[:,:,c] * (1 - alpha) + blush_overlay[:,:,c] * alpha

                # Apply Lipstick
                if st.session_state.lipstick_enabled:
                    upper_lip_points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                               for i in UPPER_LIP], np.int32)
                    lower_lip_points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                               for i in LOWER_LIP], np.int32)
                    
                    lipstick_overlay = np.zeros_like(frame)
                    cv2.fillPoly(lipstick_overlay, [upper_lip_points], hex_to_bgr(lipstick_color))
                    cv2.fillPoly(lipstick_overlay, [lower_lip_points], hex_to_bgr(lipstick_color))
                    
                    lip_mask = np.zeros_like(frame[:,:,0])
                    cv2.fillPoly(lip_mask, [upper_lip_points], 255)
                    cv2.fillPoly(lip_mask, [lower_lip_points], 255)
                    
                    alpha = (lip_mask / 255.0) * lipstick_opacity
                    for c in range(3):
                        frame[:,:,c] = frame[:,:,c] * (1 - alpha) + lipstick_overlay[:,:,c] * alpha

                # Function to apply iris color change
                def apply_iris_color(frame, iris_points, iris_color, blur_value):
                    mask = np.zeros_like(frame)
                    iris_center = np.mean(iris_points, axis=0).astype(int)
                    iris_radius = int(np.linalg.norm(iris_points[0] - iris_points[2]) // 2)

                    # Draw a filled circle for the iris color
                    cv2.circle(mask, tuple(iris_center), iris_radius, iris_color, -1)

                    # Apply Gaussian blur to smooth out the edges
                    mask = cv2.GaussianBlur(mask, (blur_value, blur_value), 0)

                    # Blend the mask with the frame
                    blended_frame = cv2.addWeighted(frame, 1, mask, 0.4, 0)
                    return blended_frame

                # Apply Iris Color
                if st.session_state.iris_enabled:
                    # Get iris points for both eyes
                    left_iris_points = np.array([
                        [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                        for i in LEFT_IRIS
                    ])
                    right_iris_points = np.array([
                        [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                        for i in RIGHT_IRIS
                    ])

                    # Apply iris color changes
                    iris_bgr = hex_to_bgr(iris_color)
                    frame = apply_iris_color(frame, left_iris_points, iris_bgr, blur_intensity)
                    frame = apply_iris_color(frame, right_iris_points, iris_bgr, blur_intensity)

                # Apply Eyeshadow
                if st.session_state.eyeshadow_enabled:
                    left_eye_points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                              for i in LEFT_EYESHADOW], np.int32)
                    right_eye_points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                               for i in RIGHT_EYESHADOW], np.int32)
                    
                    eyeshadow_overlay = np.zeros_like(frame)
                    cv2.fillPoly(eyeshadow_overlay, [left_eye_points], hex_to_bgr(eyeshadow_color))
                    cv2.fillPoly(eyeshadow_overlay, [right_eye_points], hex_to_bgr(eyeshadow_color))
                    
                    eye_mask = np.zeros_like(frame[:,:,0])
                    cv2.fillPoly(eye_mask, [left_eye_points], 255)
                    cv2.fillPoly(eye_mask, [right_eye_points], 255)
                    eye_mask = cv2.GaussianBlur(eye_mask, (15, 15), 5)
                    
                    alpha = (eye_mask / 255.0) * eyeshadow_opacity
                    for c in range(3):
                        frame[:,:,c] = frame[:,:,c] * (1 - alpha) + eyeshadow_overlay[:,:,c] * alpha

                # Apply Skin Smoothing
                if smoothing_intensity > 0:
                    face_area = cv2.bitwise_and(frame, frame, mask=face_mask)
                    smoothed = cv2.bilateralFilter(face_area, 9, smoothing_intensity, smoothing_intensity)
                    alpha_smooth = face_mask / 255.0
                    for c in range(3):
                        frame[:,:,c] = frame[:,:,c] * (1 - alpha_smooth) + smoothed[:,:,c] * alpha_smooth

                # Apply Brightness
                if brightness_increase > 0:
                    brightened = cv2.add(frame, np.ones_like(frame) * brightness_increase)
                    alpha_bright = face_mask / 255.0
                    for c in range(3):
                        frame[:,:,c] = frame[:,:,c] * (1 - alpha_bright) + brightened[:,:,c] * alpha_bright

        # Handle image capture
        if st.session_state.capture_image:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_filename = os.path.join(capture_directory, f"makeup_applied_{timestamp}.png")
            cv2.imwrite(image_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            st.success(f"✅ Image saved as {image_filename}")
            st.session_state.capture_image = False

        # Display the frame
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

else:
    st.info("👆 Enable the webcam from the sidebar to start the feed.")