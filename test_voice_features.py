import requests
import os
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_motivational_scenario():
    print("\n--- Testing GET /simulation/scenarios for Motivational Speech ---")
    try:
        resp = requests.get(f"{BASE_URL}/simulation/scenarios")
        if resp.status_code == 200:
            data = resp.json()
            if "motivational_speech" in data:
                scenario = data["motivational_speech"]
                print("✅ Motivational Commencement Speech scenario successfully found in scenarios!")
                print(f"  -> Title: {scenario['title']}")
                print(f"  -> Script length: {len(scenario['script'])} exchanges")
            else:
                print("❌ Motivational Commencement Speech scenario missing in scenarios response!")
        else:
            print(f"❌ Scenarios HTTP Failed: {resp.status_code}")
    except Exception as e:
        print(f"❌ Scenarios Error: {e}")

def test_voice_speed_and_emotion():
    print("\n--- Testing POST /process/voice with Speed and Emotion ---")
    audio_path = "test_assets/test_audio.wav"
    if not os.path.exists(audio_path):
        # Create a simple test audio if it doesn't exist
        import numpy as np
        import soundfile as sf
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.4 * np.sin(2 * np.pi * 440 * t)
        os.makedirs("test_assets", exist_ok=True)
        sf.write(audio_path, y, sr)

    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            # Test fast happy voice conversion
            resp = requests.post(
                f"{BASE_URL}/process/voice", 
                files=files, 
                params={"profile": "profile_1", "pitch_steps": 1.0, "speed_ratio": 1.5, "emotion": "happy"}
            )
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                print(f"✅ Process Voice with Speed (1.5x) and Emotion (happy) Passed!")
                print(f"  -> Latency: {data['latency_ms']}ms, SNR: {data['snr']}, MCD: {data['mcd']}")
            else:
                print(f"❌ Process Voice Failed: {data}")
        else:
            print(f"❌ Process Voice HTTP Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Process Voice Error: {e}")

def test_tts_speed_and_emotion():
    print("\n--- Testing POST /process/tts with Speed and Emotion ---")
    try:
        # Test slow sad TTS
        resp = requests.post(
            f"{BASE_URL}/process/tts", 
            params={
                "text": "The sunset was beautiful, but we had to go home.", 
                "profile": "profile_1", 
                "pitch_steps": -1.0, 
                "speed_ratio": 0.8, 
                "emotion": "sad"
            }
        )
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                print(f"✅ Process TTS with Speed (0.8x) and Emotion (sad) Passed!")
                print(f"  -> Latency: {data['latency_ms']}ms, SNR: {data['snr']}, MCD: {data['mcd']}")
            else:
                print(f"❌ Process TTS Failed: {data}")
        else:
            print(f"❌ Process TTS HTTP Failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Process TTS Error: {e}")

if __name__ == "__main__":
    print("Starting Voice speed, emotion, and scenario verification tests...")
    print(f"Connecting to backend server on {BASE_URL}...\n")
    test_motivational_scenario()
    test_voice_speed_and_emotion()
    test_tts_speed_and_emotion()
    print("\nVerification execution complete.")
