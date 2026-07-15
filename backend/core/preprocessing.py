import cv2
import numpy as np
import librosa

class Preprocessor:
    def __init__(self):
        # OpenCV Haar cascade frontal-face detector (lightweight, no extra weights).
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def normalize_audio(self, audio_path):
        """Standardizes audio to 16kHz and normalizes volume."""
        y, sr = librosa.load(audio_path, sr=16000)
        y_normalized = librosa.util.normalize(y)
        return y_normalized, sr

    def align_face(self, frame):
        """Detects and crops face with padding for GAN input."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
            
        (x, y, w, h) = faces[0]
        # Add 20% padding for better GAN blending
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        face_crop = frame[max(0, y-pad_h):y+h+pad_h, max(0, x-pad_w):x+w+pad_w]
        return cv2.resize(face_crop, (256, 256))