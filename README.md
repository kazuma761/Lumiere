# Lumiere 🌟
[![GitHub Issues](https://img.shields.io/github/issues/kazuma761/Lumiere)](https://github.com/kazuma761/Lumiere/issues)
[![Branch](https://img.shields.io/badge/branch-testing-blue)](https://github.com/kazuma761/Lumiere/tree/testing)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Lumiere is an experimental project focused on AI-powered locally run real-time video & audio enhancement plugin designed to work smoothly with platforms like Zoom, WhatsApp, Google Meet, and streaming app OBS, streamlab. It uses facial landmark detection using mediapipe and machine learning to subtly improve your appearance on video calls or streams—enhancing lighting, skin tone on your face  and sharpness & background blur while keeping your natural look intact. Unlike generic filters, it’s adaptive and professional, making it perfect for meetings, content creation, and everyday use.

## Features ✨
1) Subtle facial refinement (skin smoothing, under-eye brightening, and natural-looking contouring) to maintain a  polished appearance of the user.

2) AI powered  noise suppression tool  offers noticeable noise reduction, and  improve your voice quality in everyday environments.

3) Intelligent camera tracks , it automatically zooms and follows movement of the user.  Followered by dynamic background control (blur) with real-time.

4) Automatic face lighting adjustment that balances exposure in any environment.

5) Real-time facial feature customization equiped with adjustable intensity controllers (lip tint, cheek tone, iris color, and eye accents) having gender-neutral presets.

6) Streamlit Integration provides an interactive web interface for real-time image processing and visualization. 

**Experimental Components**:
1) MediaPipe Face Mesh
2) OpenCV 
3) TensorFlow Lite
4) Streamlit 
5) pyvirtualcam 
6) OBS Virtual Camera 
7) DirectShow Framework
## Getting Started 🚀
### Prerequisites
- Python 3.7 or higher
- Recommended: Create a virtual environment to manage dependencies.
- [OBS Studio](https://obsproject.com/) (installed and configured)
- Windows 10/11
- Webcam
- Microsoft C++ Build Tools (required for pywinpty)
 ### Installation 
 1. **Clone the repository**:
```bash
git clone -b testing https://github.com/kazuma761/Lumiere.git
cd Lumiere
```
 2. **Set up virtual environment**:

```bash
python -m venv venv
venv\Scripts\activate
```
3. **Install dependencies**:
Use the package manager [pip](https://pip.pypa.io/en/stable/).
```bash
pip install pyvirtualcam opencv-python mediapipe numpy
```
  If you encounter OpenCV issues:

```bash
pip install opencv-python-headless
pip install pydeck
```
4. **Complete installation**:

```bash
pip install -r requirements.txt
.\Zoom\setup_obs.ps1
```
### Usage
  **Launch Makeup Controls UI**:
```bash
python .\Streamlit_App\makeup_controls.py
```
  **Start Virtual Camera for Zoom**:
```bash
python .\Zoom\virtual_camera.py
```
## Troubleshooting
- Ensure OBS Virtual Camera is enabled (Tools → Virtual Camera → Start)
- Grant camera permissions in Windows Settings
- Update MediaPipe if face detection fails:
  
  ```bash
  pip install --upgrade mediapipe
  ```
- Re-run setup_obs.ps1 if virtual camera isn't detected
  
  ```bash
  .\Zoom\setup_obs.ps1
  ```
## ‪Contributing
   1. **Fork the repository**.
   2. **Create your feature branch**.
      ```bash
      git checkout -b feature/NewFeature
      ```
   3. **Commit your changes**.
      ```bash
      git commit -m 'Add new feature'
      ```
  4. **Push to the branch**.
     ```bash
      git push origin feature/NewFeature
      ```
  5. **Open a Pull Request**.

## ‪License
MIT License - See [License](LICENSE) for details

## Contact
For support, please [open an issue](https://github.com/kazuma761/Lumiere/issues) or contact [@kazuma761](https://github.com/kazuma761)

Project Link: [https://github.com/kazuma761/Lumiere](https://github.com/kazuma761/Lumiere)

