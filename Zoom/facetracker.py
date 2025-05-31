import cv2
import numpy as np
import os
from typing import Tuple, Optional
from face_tracker_config import *

class FaceTracker:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "yunet_s_640_640.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                MODEL_INPUT_SIZE,  # Initial size, will be updated in init_webcam
                SCORE_THRESHOLD,
                NMS_THRESHOLD,
                TOP_K
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load face detector model: {str(e)}")
        
        self.webcam = None
        self.input_size = MODEL_INPUT_SIZE
        
    def set_input_size(self, size: Tuple[int, int]):
        """Update the detector's input size."""
        self.input_size = size
        self.detector.setInputSize(size)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
    def init_webcam(self) -> Tuple[int, int]:
        """Initialize the webcam and return its dimensions."""
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            raise RuntimeError("Cannot access webcam")
            
        width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detector.setInputSize((width, height))
        return width, height
        
    def detect_largest_face(self, frame) -> Optional[Tuple[int, int, int, int, float]]:
        """Detect and return the largest face in the frame."""
        faces = self.detector.detect(frame)
        
        if faces[1] is None:
            return None
            
        largest_face = None
        largest_area = 0
        largest_confidence = 0
        
        for face in faces[1]:
            x, y, w, h = face[:4]
            confidence = face[-1]
            area = w * h
            
            if area > largest_area:
                largest_area = area
                largest_face = (int(x), int(y), int(w), int(h))
                largest_confidence = confidence
                
        if largest_face is not None:
            return (*largest_face, largest_confidence)
        return None
        
    def process_frame(self, frame, width: int, height: int):
        """Process a single frame and return the processed frame."""
        # Resize frame to match detector's input size if necessary
        if frame.shape[1] != self.input_size[0] or frame.shape[0] != self.input_size[1]:
            frame_resized = cv2.resize(frame, self.input_size)
            # Detect faces on resized frame
            face_data = self.detect_largest_face(frame_resized)
            # Scale coordinates back to original size if face was detected
            if face_data is not None:
                x, y, w, h, conf = face_data
                scale_x = width / self.input_size[0]
                scale_y = height / self.input_size[1]
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                face_data = (x, y, w, h, conf)
        else:
            face_data = self.detect_largest_face(frame)
            
        if face_data is None:
            return frame
            
        x, y, w, h, confidence = face_data
        
        # Add padding around the face
        startX = max(0, x - FACE_PADDING)
        startY = max(0, y - FACE_PADDING)
        endX = min(width, x + w + FACE_PADDING)
        endY = min(height, y + h + FACE_PADDING)
        
        # Draw face center point
        center_x = (startX + endX) // 2
        center_y = (startY + endY) // 2
        cv2.circle(frame, (center_x, center_y), 3, GREEN, -1)
        
        # Draw face rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), GREEN, RECTANGLE_THICKNESS)
        
        # Draw confidence
        text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   CONFIDENCE_FONT_SCALE, GREEN, CONFIDENCE_FONT_THICKNESS)
        
        # Crop and zoom
        zoomed = frame[startY:endY, startX:endX]
        if zoomed.size > 0:
            return cv2.resize(zoomed, (width, height))
        return frame
        
    def run(self):
        """Run the face tracking loop."""
        try:
            width, height = self.init_webcam()
            
            while True:
                ret, frame = self.webcam.read()
                if not ret:
                    raise RuntimeError("Cannot read from webcam")
                    
                processed_frame = self.process_frame(frame, width, height)
                cv2.imshow(WINDOW_NAME, processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during face tracking: {str(e)}")
        finally:
            self.release()
            
    def release(self):
        """Release resources."""
        if self.webcam is not None:
            self.webcam.release()
        cv2.destroyAllWindows()

def auto_frame_webcam():
    """Main function to run the face tracker."""
    try:
        with FaceTracker() as tracker:
            tracker.run()
    except Exception as e:
        print(f"Failed to initialize face tracker: {str(e)}")

if __name__ == "__main__":
    auto_frame_webcam()
