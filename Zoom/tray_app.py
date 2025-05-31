import pystray
from PIL import Image
import threading
import json
import os
from virtual_camera import VirtualMakeupCamera

class VirtualCameraApp:
    def __init__(self):
        self.virtual_cam = None
        self.cam_thread = None
        self.settings_file = "camera_settings.json"
        self.load_settings()
        
    def load_settings(self):
        """Load saved settings"""
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                self.settings = json.load(f)
        else:
            self.settings = {
                'auto_start': False,
                'makeup_enabled': True,
                'tracking_enabled': True
            }
            
    def save_settings(self):
        """Save current settings"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f)
            
    def start_camera(self):
        """Start the virtual camera in a separate thread"""
        if self.virtual_cam is None:
            self.virtual_cam = VirtualMakeupCamera()
            self.cam_thread = threading.Thread(target=self.virtual_cam.run)
            self.cam_thread.start()
            
    def stop_camera(self):
        """Stop the virtual camera"""
        if self.virtual_cam is not None:
            self.virtual_cam.release()
            self.virtual_cam = None
            if self.cam_thread:
                self.cam_thread.join()
                
    def create_tray_icon(self):
        """Create system tray icon and menu"""
        # Create a simple icon (you can replace with your own .ico file)
        icon_image = Image.new('RGB', (64, 64), color='blue')
        
        def on_clicked(icon, item):
            if str(item) == "Start Camera":
                self.start_camera()
            elif str(item) == "Stop Camera":
                self.stop_camera()
            elif str(item) == "Exit":
                self.stop_camera()
                icon.stop()
                
        # Create the menu
        menu = pystray.Menu(
            pystray.MenuItem("Start Camera", on_clicked),
            pystray.MenuItem("Stop Camera", on_clicked),
            pystray.MenuItem("Exit", on_clicked)
        )
        
        # Create the icon
        icon = pystray.Icon(
            "Virtual Makeup Camera",
            icon_image,
            "Virtual Makeup Camera",
            menu
        )
        return icon
        
    def run(self):
        """Run the application"""
        icon = self.create_tray_icon()
        icon.run()

if __name__ == "__main__":
    app = VirtualCameraApp()
    app.run()
