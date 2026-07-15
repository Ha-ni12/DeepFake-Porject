import requests
import io
import time

base_url = "http://127.0.0.1:8000"

print("Checking Health")
try:
    health = requests.get(base_url + "/health")
    print("Health:", health.status_code, health.text)
except Exception as e:
    print("Error:", e)
    
print("Checking TTS")
try:
    response = requests.post(base_url + "/process/tts", params={"text": "Hello world", "profile": "profile_1", "pitch_steps": 2.0})
    print("TTS:", response.status_code, response.text[:200])
except Exception as e:
    print("Error:", e)
    
print("Checking Voice Upload")
import numpy as np
import soundfile as sf
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    # generate random noise
    sr = 16000
    noise = np.random.randn(sr * 1).astype(np.float32)
    sf.write(f.name, noise, sr)
    f.flush()
    try:
        with open(f.name, "rb") as audio_file:
            files = {"file": audio_file}
            response = requests.post(base_url + "/process/voice", files=files, params={"pitch_steps": 2.0})
            print("Voice:", response.status_code, response.text[:200])
    except Exception as e:
        print("Error:", e)
os.remove(f.name)
