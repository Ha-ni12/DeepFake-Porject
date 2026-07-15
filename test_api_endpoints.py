import requests
import io
import time
import os
import sys

# Optional: suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import cv2
    import soundfile as sf
except ImportError:
    print("Please install requirements: pip install numpy opencv-python soundfile requests")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8000"

def create_test_assets():
    os.makedirs("test_assets", exist_ok=True)
    
    # 1. Create Audio (2s sine wave)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 440 * t)
    audio_path = "test_assets/test_audio.wav"
    sf.write(audio_path, y, sr)
    
    # 2. Create Image (Dummy Face)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 50, (200, 150, 100), -1)
    img_path = "test_assets/test_face.jpg"
    cv2.imwrite(img_path, img)
    
    return audio_path, img_path

def test_health():
    print("\n--- Testing GET /health ---")
    resp = requests.get(f"{BASE_URL}/health")
    if resp.status_code == 200:
        print("✅ Health Check Passed")
    else:
        print(f"❌ Health Check Failed: {resp.status_code} - {resp.text}")

def test_process_frame(img_path):
    print("\n--- Testing POST /process/frame ---")
    try:
        with open(img_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(f"{BASE_URL}/process/frame", files=files, params={"profile": "profile_1"})
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                print(f"✅ Process Frame Passed (Latency: {data['latency_ms']}ms, SSIM: {data['ssim']})")
                
                # Check output image
                img_data = data["image_data"]
                if img_data:
                    print("  -> Base64 Image returned successfully.")
            else:
                 print(f"❌ Process Frame Failed: {data}")
        else:
            print(f"❌ Process Frame HTTP Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
         print(f"❌ Process Frame Error: {e}")

def test_process_voice(audio_path):
    print("\n--- Testing POST /process/voice ---")
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(f"{BASE_URL}/process/voice", files=files, params={"profile": "profile_1", "pitch_steps": 2.0})
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                print(f"✅ Process Voice Passed (Latency: {data['latency_ms']}ms, SNR: {data['snr']})")
            else:
                 print(f"❌ Process Voice Failed: {data}")
        else:
            print(f"❌ Process Voice HTTP Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
         print(f"❌ Process Voice Error: {e}")

def test_process_tts():
    print("\n--- Testing POST /process/tts ---")
    try:
        resp = requests.post(
            f"{BASE_URL}/process/tts", 
            params={"text": "Hello this is a test.", "profile": "profile_1"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                print(f"✅ Process TTS Passed (Latency: {data['latency_ms']}ms)")
            else:
                 print(f"❌ Process TTS Failed: {data}")
        else:
            print(f"❌ Process TTS HTTP Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
         print(f"❌ Process TTS Error: {e}")

def test_scenarios():
    print("\n--- Testing GET /simulation/scenarios ---")
    try:
        resp = requests.get(f"{BASE_URL}/simulation/scenarios")
        if resp.status_code == 200:
            data = resp.json()
            if "comedy_interview" in data:
                 print(f"✅ Scenarios Retrieved successfully ({len(data)} scenarios found)")
            else:
                 print(f"❌ Scenarios structure incorrect: {data}")
        else:
            print(f"❌ Scenarios HTTP Failed: {resp.status_code}")
    except Exception as e:
         print(f"❌ Scenarios Error: {e}")

def test_chat():
    print("\n--- Testing POST /conversation/chat ---")
    try:
        resp = requests.post(
            f"{BASE_URL}/conversation/chat",
            params={"message": "Are you a test script?", "profile": "profile_1"}
        )
        if resp.status_code == 200:
            data = resp.json()
            if "reply" in data:
                 print(f"✅ Chat Passed - AI Reply: '{data['reply'][:50]}...'")
            else:
                 print(f"❌ Chat format incorrect: {data}")
        else:
            print(f"❌ Chat HTTP Failed: {resp.status_code}")
    except Exception as e:
         print(f"❌ Chat Error: {e}")

def test_export_report():
    print("\n--- Testing GET /export/report ---")
    try:
        resp = requests.get(f"{BASE_URL}/export/report")
        if resp.status_code == 200:
            print("✅ Export Report Passed (Received CSV Data)")
            sample_content = resp.text.split('\n')[:2]
            print(f"  -> File header: {sample_content[0]}")
        else:
            print(f"❌ Export Report Failed: {resp.status_code} - Run some processing first to generate the report.")
    except Exception as e:
         print(f"❌ Export Report Error: {e}")

if __name__ == "__main__":
    print("Preparing test assets...")
    audio_path, img_path = create_test_assets()
    
    print("\nStarting Automated Testing Suite...")
    print("Ensure the server is running on http://127.0.0.1:8000 using 'start_server.bat'.\n")
    
    test_health()
    test_process_tts()
    test_process_voice(audio_path)
    test_process_frame(img_path)
    test_scenarios()
    test_chat()
    test_export_report()
    
    print("\n✅ All endpoint tests executed format completed.")
