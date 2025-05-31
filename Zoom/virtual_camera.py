import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
from typing import Optional, Tuple
from face_tracker_config import *
from facetracker import FaceTracker

class VirtualMakeupCamera:
    def __init__(self):
        # Initialize face tracker
        self.face_tracker = FaceTracker()
        
        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        
        # Initialize camera
        self.camera = None
        self.virtual_cam = None
        
        # Makeup settings
        self.makeup_settings = {
            'blush_enabled': False,
            'lipstick_enabled': False,
            'eyeshadow_enabled': False,
            'iris_enabled': False,
            'smoothing_enabled': False
        }
        
    def init_cameras(self):
        """Initialize physical and virtual cameras"""
        # Initialize physical camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot access physical webcam")
            
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        
        # Set input size for face detector
        self.face_tracker.set_input_size((width, height))
        
        # Try to create virtual camera specifically for OBS
        try:
            # Ensure we're using the exact name that OBS expects
            self.virtual_cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                backend="obs",  # Explicitly use OBS backend
                device="OBS Virtual Camera",  # This must match OBS's device name exactly
                fmt=pyvirtualcam.PixelFormat.BGR  # Use BGR format to avoid conversion
            )
            print(f"Successfully created virtual camera: {self.virtual_cam.device}")
        except Exception as e:
            print(f"Error creating virtual camera: {str(e)}")
            print("\nPlease ensure that:")
            print("1. OBS Studio is installed")
            print("2. OBS Virtual Camera is started from OBS Studio")
            print("3. No other application is using the virtual camera")
            raise RuntimeError("Failed to initialize virtual camera")
            
        return width, height
        
    def process_frame(self, frame):
        """Process frame with both face tracking and makeup effects"""
        # First apply face tracking (auto-framing)
        if self.face_tracker:
            frame = self.face_tracker.process_frame(frame, frame.shape[1], frame.shape[0])
            
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            # Apply makeup effects here (to be implemented)
            pass
            
        return frame
        
    def run(self):
        """Run the virtual camera"""
        try:
            width, height = self.init_cameras()
            print("Virtual camera is running. Press Ctrl+C to stop.")
            
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                    
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Send to virtual camera (no need to convert to RGB since we're using BGR format)
                self.virtual_cam.send(processed_frame)
                self.virtual_cam.sleep_until_next_frame()
                
        except KeyboardInterrupt:
            print("\nStopping virtual camera...")
        except Exception as e:
            print(f"Error in virtual camera: {str(e)}")
        finally:
            self.release()
            
    def release(self):
        """Release all resources"""
        if self.camera is not None:
            self.camera.release()
        if self.virtual_cam is not None:
            self.virtual_cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_cam = VirtualMakeupCamera()
    virtual_cam.run()
