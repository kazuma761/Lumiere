"""Configuration settings for the face tracker."""

# Model settings
MODEL_INPUT_SIZE = (320, 320)
SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Webcam settings
FACE_PADDING = 50
WINDOW_NAME = "Auto Frame"
CONFIDENCE_FONT_SCALE = 0.7
CONFIDENCE_FONT_THICKNESS = 2
RECTANGLE_THICKNESS = 2

# Colors (BGR format)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
