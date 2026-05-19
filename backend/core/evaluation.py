import numpy as np
import cv2
import librosa
import csv
import os
import threading
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

class Evaluator:
    def __init__(self, log_file="session_metrics.csv"):
        self.log_file = log_file
        # Serialise concurrent CSV writes (FastAPI is async / multi-threaded)
        self._lock = threading.Lock()
        # Initialize the CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Type", "SSIM", "PSNR", "SNR_dB", "MCD", "Latency_ms"])

    def log_to_csv(self, m_type, ssim="-", psnr="-", snr="-", mcd="-", latency=0):
        """Logs a single processing event to the persistent CSV file."""
        with self._lock, open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                m_type, ssim, psnr, snr, mcd, f"{latency:.2f}"
            ])

    @staticmethod
    def calculate_snr(original, generated):
        min_len = min(len(original), len(generated))
        orig = original[:min_len]
        gen = generated[:min_len]
        noise = orig - gen
        signal_power = np.sum(orig**2)
        noise_power = np.sum(noise**2)
        if noise_power == 0: return 100.0
        return 10 * np.log10(signal_power / (noise_power + 1e-10))

    @staticmethod
    def calculate_mcd(original_audio, generated_audio, sr=16000):
        mfcc_orig = librosa.feature.mfcc(y=original_audio, sr=sr, n_mfcc=13)
        mfcc_gen = librosa.feature.mfcc(y=generated_audio, sr=sr, n_mfcc=13)
        min_frames = min(mfcc_orig.shape[1], mfcc_gen.shape[1])
        mfcc_orig = mfcc_orig[:, :min_frames]
        mfcc_gen = mfcc_gen[:, :min_frames]
        diff = mfcc_orig - mfcc_gen
        mcd_dist = np.mean(np.sqrt(np.sum(diff**2, axis=0)))
        return (10 / np.log(10)) * np.sqrt(2) * mcd_dist

    @staticmethod
    def calculate_psnr(original_img, generated_img):
        # Cast to float32 to avoid uint8 wrap-around in subtraction
        a = original_img.astype(np.float32)
        b = generated_img.astype(np.float32)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        mse = np.mean((a - b) ** 2)
        if mse == 0: return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))

    @staticmethod
    def calculate_ssim(original_img, generated_img):
        # Ensure matching shapes before SSIM
        if original_img.shape != generated_img.shape:
            generated_img = cv2.resize(
                generated_img,
                (original_img.shape[1], original_img.shape[0])
            )
        gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)
        # Win size must be <= smaller dim and odd; clamp for tiny crops
        min_side = min(gray_orig.shape[:2])
        win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win < 3:
            return 1.0
        score, _ = ssim(gray_orig, gray_gen, full=True, win_size=win)
        return score