import os
import sys

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.dsp_models.voice_cloning import VoiceCloner
from backend.core.watermarking import apply_audio_watermark
from backend.core.evaluation import Evaluator

import numpy as np
import librosa
import soundfile as sf
import tempfile
import time

cloner = VoiceCloner(sample_rate=16000)
evaluator = Evaluator(log_file="test_metrics.csv")

def test_voice():
    # 1. Create dummy audio
    sr = 16000
    y = np.random.randn(sr * 1).astype(np.float32)

    # 2. Pitch shift
    try:
        print("Testing pitch shift...")
        y_shifted = cloner.pitch_shift(y, n_steps=2.0)
    except Exception as e:
        print("Error in pitch shift:", e)
        return

    # 3. Apply watermark
    try:
        print("Testing watermarking...")
        y_watermarked = apply_audio_watermark(y_shifted)
    except Exception as e:
        print("Error in watermarking:", e)
        return

    # 4. Evaluate SNR and MCD
    try:
        print("Testing SNR...")
        snr_score = evaluator.calculate_snr(y, y_watermarked)
        print("SNR:", snr_score)
    except Exception as e:
        print("Error in SNR:", e)
        import traceback
        traceback.print_exc()
        
    try:
        print("Testing MCD...")
        mcd_score = evaluator.calculate_mcd(y, y_watermarked)
        print("MCD:", mcd_score)
    except Exception as e:
        print("Error in MCD:", e)
        import traceback
        traceback.print_exc()

test_voice()
