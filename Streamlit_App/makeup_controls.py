import tkinter as tk
from tkinter import ttk, colorchooser
import json
import os

class MakeupControlPanel:
    def __init__(self, virtual_camera):
        self.virtual_camera = virtual_camera
        self.root = tk.Tk()
        self.root.title("Makeup Controls")
        self.root.geometry("300x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both')
        
        # Create tabs
        self.create_blush_tab()
        self.create_lipstick_tab()
        self.create_eyes_tab()
        self.create_skin_tab()
        
        # Save/Load settings
        self.create_settings_frame()
        
    def create_blush_tab(self):
        blush_frame = ttk.Frame(self.notebook)
        self.notebook.add(blush_frame, text='Blush')
        
        # Blush controls
        ttk.Checkbutton(blush_frame, text="Enable Blush", 
                       command=lambda: self.toggle_effect('blush_enabled')).pack(pady=5)
        
        ttk.Button(blush_frame, text="Choose Blush Color", 
                  command=lambda: self.choose_color('blush_color')).pack(pady=5)
        
        ttk.Label(blush_frame, text="Blush Opacity").pack(pady=5)
        self.create_slider(blush_frame, 'blush_opacity', 0.1, 1.0, 0.2)
        
    def create_lipstick_tab(self):
        lipstick_frame = ttk.Frame(self.notebook)
        self.notebook.add(lipstick_frame, text='Lipstick')
        
        ttk.Checkbutton(lipstick_frame, text="Enable Lipstick", 
                       command=lambda: self.toggle_effect('lipstick_enabled')).pack(pady=5)
        
        ttk.Button(lipstick_frame, text="Choose Lipstick Color", 
                  command=lambda: self.choose_color('lipstick_color')).pack(pady=5)
        
        ttk.Label(lipstick_frame, text="Lipstick Opacity").pack(pady=5)
        self.create_slider(lipstick_frame, 'lipstick_opacity', 0.1, 0.5, 0.13)
        
    def create_eyes_tab(self):
        eyes_frame = ttk.Frame(self.notebook)
        self.notebook.add(eyes_frame, text='Eyes')
        
        # Iris controls
        ttk.Label(eyes_frame, text="Iris Settings").pack(pady=5)
        ttk.Checkbutton(eyes_frame, text="Enable Iris Color", 
                       command=lambda: self.toggle_effect('iris_enabled')).pack()
        ttk.Button(eyes_frame, text="Choose Iris Color", 
                  command=lambda: self.choose_color('iris_color')).pack(pady=5)
        
        ttk.Label(eyes_frame, text="Blur Intensity").pack(pady=5)
        self.create_slider(eyes_frame, 'blur_intensity', 1, 25, 21, True)
        
        # Eyeshadow controls
        ttk.Label(eyes_frame, text="\nEyeshadow Settings").pack(pady=5)
        ttk.Checkbutton(eyes_frame, text="Enable Eyeshadow", 
                       command=lambda: self.toggle_effect('eyeshadow_enabled')).pack()
        ttk.Button(eyes_frame, text="Choose Eyeshadow Color", 
                  command=lambda: self.choose_color('eyeshadow_color')).pack(pady=5)
        
        ttk.Label(eyes_frame, text="Eyeshadow Opacity").pack(pady=5)
        self.create_slider(eyes_frame, 'eyeshadow_opacity', 0.1, 1.0, 0.14)
        
    def create_skin_tab(self):
        skin_frame = ttk.Frame(self.notebook)
        self.notebook.add(skin_frame, text='Skin')
        
        ttk.Checkbutton(skin_frame, text="Enable Skin Smoothing", 
                       command=lambda: self.toggle_effect('smoothing_enabled')).pack(pady=5)
        
        ttk.Label(skin_frame, text="Smoothing Intensity").pack(pady=5)
        self.create_slider(skin_frame, 'smoothing_intensity', 0, 100, 15, True)
        
        ttk.Label(skin_frame, text="Brightness").pack(pady=5)
        self.create_slider(skin_frame, 'brightness_increase', 0, 100, 4, True)
        
    def create_settings_frame(self):
        settings_frame = ttk.Frame(self.root)
        settings_frame.pack(fill='x', pady=10)
        
        ttk.Button(settings_frame, text="Save Settings", 
                  command=self.save_settings).pack(side='left', padx=5)
        ttk.Button(settings_frame, text="Load Settings", 
                  command=self.load_settings).pack(side='right', padx=5)
        
    def create_slider(self, parent, setting_name, min_val, max_val, default, is_int=False):
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient='horizontal',
                         command=lambda v: self.update_setting(setting_name, v, is_int))
        slider.set(default)
        slider.pack(fill='x', padx=20)
        
    def toggle_effect(self, setting_name):
        current = self.virtual_camera.makeup_settings[setting_name]
        self.virtual_camera.makeup_settings[setting_name] = not current
        
    def choose_color(self, setting_name):
        color = colorchooser.askcolor(color=self.virtual_camera.makeup_settings[setting_name])[1]
        if color:
            self.virtual_camera.makeup_settings[setting_name] = color
            
    def update_setting(self, setting_name, value, is_int=False):
        if is_int:
            value = int(float(value))
        else:
            value = float(value)
        self.virtual_camera.makeup_settings[setting_name] = value
        
    def save_settings(self):
        settings_file = "makeup_settings.json"
        with open(settings_file, 'w') as f:
            json.dump(self.virtual_camera.makeup_settings, f, indent=4)
            
    def load_settings(self):
        settings_file = "makeup_settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                self.virtual_camera.makeup_settings.update(settings)
                
    def run(self):
        self.root.mainloop()

# Update VirtualMakeupCamera class to work with the control panel
def run_with_controls():
    from virtual_makeup_camera import VirtualMakeupCamera
    import threading
    
    virtual_cam = VirtualMakeupCamera()
    
    # Run virtual camera in a separate thread
    camera_thread = threading.Thread(target=virtual_cam.run)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Create and run control panel
    control_panel = MakeupControlPanel(virtual_cam)
    control_panel.run()
    
    # Cleanup
    virtual_cam.release()

if __name__ == "__main__":
    run_with_controls()
