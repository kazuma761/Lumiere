import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
import os
import time
from typing import Optional, Tuple

class VirtualMakeupCamera:
    def __init__(self):
        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        
        # Initialize cameras
        self.camera = None
        self.virtual_cam = None
        
        # Facial Landmarks Definitions
        self.LEFT_CHEEK_CONTOUR = [116, 123, 147, 192, 176, 149, 150, 136, 172, 58, 132]
        self.RIGHT_CHEEK_CONTOUR = [345, 352, 376, 421, 405, 378, 379, 365, 397, 288, 361]
        self.UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 312, 13, 82, 81, 80, 191, 78]
        self.LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 402, 317, 14, 87, 178, 88, 95, 78, 61]
        self.LEFT_EYESHADOW = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
        self.RIGHT_EYESHADOW = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Makeup settings with default values
        self.makeup_settings = {
            'blush_enabled': True,
            'blush_color': "#BB704C",
            'blush_opacity': 0.2,
            
            'lipstick_enabled': True,
            'lipstick_color': "#A44D4D",
            'lipstick_opacity': 0.13,
            
            'eyeshadow_enabled': True,
            'eyeshadow_color': "#7A4B3A",
            'eyeshadow_opacity': 0.14,
            
            'iris_enabled': True,
            'iris_color': "#060D23",
            'blur_intensity': 21,
            
            'smoothing_enabled': True,
            'smoothing_intensity': 15,
            'brightness_increase': 4
        }
        
    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)
        
    def ensure_odd(self, num):
        """Ensure number is odd and positive"""
        num = max(1, int(num))
        return num + 1 if num % 2 == 0 else num
        
    def apply_makeup(self, frame, face_landmarks):
        h, w, _ = frame.shape
        
        # Create face mask
        face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        face_points = np.array([[int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)] 
                               for i in self.FACE_OVAL], np.int32)
        cv2.fillPoly(face_mask, [face_points], 255)
        
        # Apply Blush
        if self.makeup_settings['blush_enabled']:
            face_width = abs(int(face_landmarks.landmark[454].x * w) - int(face_landmarks.landmark[234].x * w))
            left_blush_center = np.array([
                int(face_landmarks.landmark[116].x * w),
                int((face_landmarks.landmark[123].y + face_landmarks.landmark[147].y) * h / 2)
            ])
            right_blush_center = np.array([
                int(face_landmarks.landmark[345].x * w),
                int((face_landmarks.landmark[352].y + face_landmarks.landmark[376].y) * h / 2)
            ])
            
            blush_overlay = np.zeros_like(frame)
            radius = int(face_width * 0.15)
            
            cv2.circle(blush_overlay, tuple(left_blush_center), radius, 
                      self.hex_to_bgr(self.makeup_settings['blush_color']), -1)
            cv2.circle(blush_overlay, tuple(right_blush_center), radius, 
                      self.hex_to_bgr(self.makeup_settings['blush_color']), -1)
            
            # Ensure kernel size is odd and positive
            kernel_size = self.ensure_odd(55)
            blush_overlay = cv2.GaussianBlur(blush_overlay, (kernel_size, kernel_size), 20)
            
            blush_mask = np.zeros_like(frame[:,:,0])
            cv2.circle(blush_mask, tuple(left_blush_center), radius, 255, -1)
            cv2.circle(blush_mask, tuple(right_blush_center), radius, 255, -1)
            blush_mask = cv2.GaussianBlur(blush_mask, (kernel_size, kernel_size), 20)
            
            alpha = (blush_mask / 255.0) * self.makeup_settings['blush_opacity'] * 0.8
            for c in range(3):
                frame[:,:,c] = frame[:,:,c] * (1 - alpha) + blush_overlay[:,:,c] * alpha
        
        # Apply Lipstick
        if self.makeup_settings['lipstick_enabled']:
            upper_lip_points = np.array([[int(face_landmarks.landmark[i].x * w), 
                                        int(face_landmarks.landmark[i].y * h)]
                                       for i in self.UPPER_LIP], np.int32)
            lower_lip_points = np.array([[int(face_landmarks.landmark[i].x * w), 
                                        int(face_landmarks.landmark[i].y * h)]
                                       for i in self.LOWER_LIP], np.int32)
            
            lipstick_overlay = np.zeros_like(frame)
            cv2.fillPoly(lipstick_overlay, [upper_lip_points], 
                        self.hex_to_bgr(self.makeup_settings['lipstick_color']))
            cv2.fillPoly(lipstick_overlay, [lower_lip_points], 
                        self.hex_to_bgr(self.makeup_settings['lipstick_color']))
            
            lip_mask = np.zeros_like(frame[:,:,0])
            cv2.fillPoly(lip_mask, [upper_lip_points], 255)
            cv2.fillPoly(lip_mask, [lower_lip_points], 255)
            
            alpha = (lip_mask / 255.0) * self.makeup_settings['lipstick_opacity']
            for c in range(3):
                frame[:,:,c] = frame[:,:,c] * (1 - alpha) + lipstick_overlay[:,:,c] * alpha
        
        # Apply Iris Color
        if self.makeup_settings['iris_enabled']:
            def apply_iris_color(frame, iris_points, iris_color, blur_value):
                mask = np.zeros_like(frame)
                iris_center = np.mean(iris_points, axis=0).astype(int)
                iris_radius = int(np.linalg.norm(iris_points[0] - iris_points[2]) // 2)
                cv2.circle(mask, tuple(iris_center), iris_radius, iris_color, -1)
                # Ensure blur value is odd and positive
                blur_value = self.ensure_odd(blur_value)
                mask = cv2.GaussianBlur(mask, (blur_value, blur_value), 0)
                return cv2.addWeighted(frame, 1, mask, 0.4, 0)

            left_iris_points = np.array([[int(face_landmarks.landmark[i].x * w), 
                                        int(face_landmarks.landmark[i].y * h)]
                                       for i in self.LEFT_IRIS])
            right_iris_points = np.array([[int(face_landmarks.landmark[i].x * w), 
                                         int(face_landmarks.landmark[i].y * h)]
                                        for i in self.RIGHT_IRIS])

            iris_bgr = self.hex_to_bgr(self.makeup_settings['iris_color'])
            frame = apply_iris_color(frame, left_iris_points, iris_bgr, 
                                   self.makeup_settings['blur_intensity'])
            frame = apply_iris_color(frame, right_iris_points, iris_bgr, 
                                   self.makeup_settings['blur_intensity'])
        
        # Apply Eyeshadow
        if self.makeup_settings['eyeshadow_enabled']:
            left_eye_points = np.array([[int(face_landmarks.landmark[i].x * w), 
                                       int(face_landmarks.landmark[i].y * h)]
                                      for i in self.LEFT_EYESHADOW], np.int32)
            right_eye_points = np.array([[int(face_landmarks.landmark[i].x * w), 
                                        int(face_landmarks.landmark[i].y * h)]
                                       for i in self.RIGHT_EYESHADOW], np.int32)
            
            eyeshadow_overlay = np.zeros_like(frame)
            cv2.fillPoly(eyeshadow_overlay, [left_eye_points], 
                        self.hex_to_bgr(self.makeup_settings['eyeshadow_color']))
            cv2.fillPoly(eyeshadow_overlay, [right_eye_points], 
                        self.hex_to_bgr(self.makeup_settings['eyeshadow_color']))
            
            eye_mask = np.zeros_like(frame[:,:,0])
            cv2.fillPoly(eye_mask, [left_eye_points], 255)
            cv2.fillPoly(eye_mask, [right_eye_points], 255)
            # Ensure kernel size is odd and positive
            eye_kernel_size = self.ensure_odd(15)
            eye_mask = cv2.GaussianBlur(eye_mask, (eye_kernel_size, eye_kernel_size), 5)
            
            alpha = (eye_mask / 255.0) * self.makeup_settings['eyeshadow_opacity']
            for c in range(3):
                frame[:,:,c] = frame[:,:,c] * (1 - alpha) + eyeshadow_overlay[:,:,c] * alpha
        
        # Apply Skin Smoothing
        if self.makeup_settings['smoothing_enabled']:
            face_area = cv2.bitwise_and(frame, frame, mask=face_mask)
            smoothed = cv2.bilateralFilter(face_area, 9, 
                                         self.makeup_settings['smoothing_intensity'],
                                         self.makeup_settings['smoothing_intensity'])
            alpha_smooth = face_mask / 255.0
            for c in range(3):
                frame[:,:,c] = frame[:,:,c] * (1 - alpha_smooth) + smoothed[:,:,c] * alpha_smooth

            # Apply Brightness
            if self.makeup_settings['brightness_increase'] > 0:
                brightened = cv2.add(frame, 
                                   np.ones_like(frame) * self.makeup_settings['brightness_increase'])
                alpha_bright = face_mask / 255.0
                for c in range(3):
                    frame[:,:,c] = frame[:,:,c] * (1 - alpha_bright) + brightened[:,:,c] * alpha_bright
        
        return frame
        
    def init_cameras(self):
        """Initialize physical and virtual cameras"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot access physical webcam")
            
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        
        try:
            self.virtual_cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=fps,
                backend="obs",
                device="OBS Virtual Camera",
                fmt=pyvirtualcam.PixelFormat.BGR
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
        """Process frame with makeup effects"""
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = self.apply_makeup(frame, face_landmarks)
                
        return frame
        
    def run(self):
        """Run the virtual makeup camera"""
        try:
            width, height = self.init_cameras()
            print("Virtual makeup camera is running. Press Ctrl+C to stop.")
            
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                    
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Send to virtual camera
                self.virtual_cam.send(processed_frame)
                self.virtual_cam.sleep_until_next_frame()
                
        except KeyboardInterrupt:
            print("\nStopping virtual makeup camera...")
        except Exception as e:
            print(f"Error in virtual makeup camera: {str(e)}")
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
