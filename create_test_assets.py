"""
create_test_assets.py — Creates test audio and image files for functional testing.
Run from project root: python create_test_assets.py
"""
import numpy as np
import os

# ── Test audio: 2-second 440 Hz sine wave (A4 note) at 16 kHz ──────
try:
    import soundfile as sf
    sr = 16000
    duration = 2.0
    t  = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y  = 0.4 * np.sin(2 * np.pi * 440 * t)   # 440 Hz sine
    y += 0.15 * np.sin(2 * np.pi * 880 * t)   # add harmonic
    os.makedirs("test_assets", exist_ok=True)
    sf.write("test_assets/test_audio.wav", y, sr)
    print("✓ Created test_assets/test_audio.wav (2 s, 440 Hz sine)")
except ImportError:
    print("✗ soundfile not installed — skipping audio")

# ── Test image: synthetic face-like portrait ───────────────────────
try:
    import cv2
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Background gradient
    for i in range(256):
        img[i, :] = [max(0, 60 - i // 4), max(0, 40 - i // 5), max(0, 80 - i // 4)]

    # Skin-tone oval (face)
    cv2.ellipse(img, (128, 130), (70, 90), 0, 0, 360, (180, 140, 100), -1)

    # Eyes
    cv2.ellipse(img, (100, 110), (18, 12), 0, 0, 360, (40, 30, 20), -1)
    cv2.ellipse(img, (156, 110), (18, 12), 0, 0, 360, (40, 30, 20), -1)
    cv2.circle(img, (100, 110), 8,  (10, 8, 5),  -1)
    cv2.circle(img, (156, 110), 8,  (10, 8, 5),  -1)
    cv2.circle(img, (103, 107), 3,  (240, 240, 240), -1)
    cv2.circle(img, (159, 107), 3,  (240, 240, 240), -1)

    # Nose
    pts = np.array([[128, 130], [120, 158], [136, 158]], np.int32)
    cv2.polylines(img, [pts], True, (140, 100, 70), 2)

    # Mouth / smile
    cv2.ellipse(img, (128, 175), (28, 14), 0, 0, 180, (130, 60, 60), 2)

    # Eyebrows
    cv2.line(img, (84, 92), (116, 88), (60, 40, 20), 3)
    cv2.line(img, (140, 88), (172, 92), (60, 40, 20), 3)

    # Ears
    cv2.ellipse(img, (56,  130), (14, 22), 0, 0, 360, (165, 125, 90), -1)
    cv2.ellipse(img, (200, 130), (14, 22), 0, 0, 360, (165, 125, 90), -1)

    # Hair
    cv2.ellipse(img, (128, 60), (75, 55), 0, 0, 360, (30, 20, 10), -1)
    cv2.rectangle(img, (54, 60), (202, 80), (30, 20, 10), -1)

    os.makedirs("test_assets", exist_ok=True)
    cv2.imwrite("test_assets/test_face.jpg", img)
    print("✓ Created test_assets/test_face.jpg (256×256 synthetic face)")
except ImportError:
    print("✗ opencv not installed — skipping image")

print("\nDone. Use these files to test the Voice Cloning and Face Swap tabs.")
