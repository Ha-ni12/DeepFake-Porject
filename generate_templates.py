"""
generate_templates.py — Creates placeholder template images for face swap profiles.
Run once from the project root: python generate_templates.py
"""
import cv2
import numpy as np
import os

os.makedirs("backend/dsp_models/templates", exist_ok=True)

def make_placeholder(filename, color_bgr, label):
    img = np.full((256, 256, 3), color_bgr, dtype=np.uint8)
    cv2.putText(img, label, (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(img, "Template", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imwrite(filename, img)
    print(f"Created: {filename}")

make_placeholder("backend/dsp_models/templates/profile_1.jpg", (60, 40, 100),  "Public Figure A")
make_placeholder("backend/dsp_models/templates/profile_2.jpg", (40, 80, 110),  "Public Figure B")
print("Done.")
