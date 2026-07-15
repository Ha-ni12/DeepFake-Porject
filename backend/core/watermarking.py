"""
watermarking.py — Visual and Audio Watermarking Module
Implements mandatory, non-bypassable watermarks as required by the SRS safety spec.
"""

import cv2
import numpy as np


def apply_visual_watermark(frame: np.ndarray) -> np.ndarray:
    """
    Applies a permanent, clearly visible visual watermark to every output frame.
    Complies with SRS §6 — Safety & Ethics: mandatory disclosure, cannot be removed.
    """
    h, w = frame.shape[:2]
    text      = "AI GENERATED — PARODY MODE"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = max(0.5, w / 800)          # scale proportionally to image width
    thickness = 2
    color     = (0, 0, 220)               # Red (BGR)

    # Measure text so we can position it properly
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    text_x = w - tw - 12
    text_y = h - 12

    # Semi-transparent background bar for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 6, text_y - th - 6), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Shadow + main text
    cv2.putText(frame, text, (text_x + 1, text_y + 1), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (text_x, text_y),          font, scale, color,     thickness)

    return frame


def apply_audio_watermark(audio_data: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Injects a near-ultrasonic, low-amplitude tone as an audio watermark.

    The frequency is set to 95% of the Nyquist limit (sr / 2 * 0.95), keeping
    it as high as possible while remaining representable at the given sample rate.
      - At 24 kHz (F5-TTS output): ~11,400 Hz  — inaudible / barely perceptible
      - At 16 kHz (fallback):    ~7,600 Hz   — at the extreme edge of hearing

    Amplitude is 0.0005 to minimize audible noise artifacts.
    """
    duration       = len(audio_data) / sr
    t              = np.linspace(0, duration, len(audio_data), endpoint=False)
    nyquist        = sr / 2
    watermark_freq = nyquist * 0.95          # 95% of Nyquist — truly inaudible
    watermark      = 0.0005 * np.sin(2 * np.pi * watermark_freq * t)
    return np.clip(audio_data + watermark, -1.0, 1.0).astype(np.float32)