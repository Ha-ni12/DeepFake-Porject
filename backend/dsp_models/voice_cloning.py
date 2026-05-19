"""
voice_cloning.py — Voice Cloning Module using Coqui XTTS v2
Performs high-quality zero-shot voice cloning for both TTS and audio-to-audio.
Supports preset voice profiles and user-uploaded target voice samples.
"""

import io
import os
import re
import tempfile
import time
import threading
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
import soundfile as sf
# pyrefly: ignore [missing-import]
import torch

# ── Compatibility patch for coqui-tts + transformers 5.x ────────────
# coqui-tts imports `isin_mps_friendly` from transformers, which was
# removed in transformers 5.x. On Windows/CUDA it's just torch.isin.
try:
    # pyrefly: ignore [missing-import]
    import transformers.pytorch_utils as _pt_utils
    if not hasattr(_pt_utils, 'isin_mps_friendly'):
        _pt_utils.isin_mps_friendly = torch.isin
except Exception:
    pass

# ── torchaudio.load monkey-patch ─────────────────────────────────────
# Recent torchaudio versions route ALL audio loading through torchcodec,
# which requires FFmpeg DLLs that are not always available on Windows.
# XTTS v2 calls torchaudio.load() internally to read the reference WAV.
# We replace it with a soundfile-based implementation that works without
# any FFmpeg dependency.
try:
    # pyrefly: ignore [missing-import]
    import torchaudio as _torchaudio

    def _sf_load(filepath, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, format=None, backend=None,
                 encoding=None, bits_per_sample=None):
        """soundfile-backed replacement for torchaudio.load (no FFmpeg needed)."""
        data, sr = sf.read(str(filepath), dtype='float32', always_2d=True)
        # data shape: (samples, channels) — convert to (channels, samples)
        tensor = torch.from_numpy(data.T)
        if frame_offset:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        return tensor, sr

    _torchaudio.load = _sf_load
    print("[VoiceCloner] torchaudio.load patched → soundfile backend (no FFmpeg required).")
except Exception as _e:
    print(f"[VoiceCloner] WARNING: Could not patch torchaudio.load: {_e}")

# ── Constants ────────────────────────────────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(__file__))
_TEMPLATES_DIR = os.path.join(_MODEL_DIR, "templates")

# Default voice reference files for built-in profiles
# Place short WAV clips (3-10 seconds of clean speech) in the templates folder.
_PROFILE_VOICE_MAP = {
    "profile_1": os.path.join(_TEMPLATES_DIR, "profile_1_voice.wav"),
    "profile_2": os.path.join(_TEMPLATES_DIR, "profile_2_voice.wav"),
}


class VoiceCloner:
    def __init__(self, sample_rate=24000, vram_timeout=10.0):
        """
        Initialises the Coqui XTTS v2 model for zero-shot voice cloning.
        Falls back to basic DSP pitch shifting if Coqui TTS is unavailable.
        """
        self.sr = sample_rate
        self.tts_engine = None
        self._whisper_model = None  # lazy-loaded; cached after first transcription
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Pre-compile regex for performance
        self._bracket_regex = re.compile(r'[\(\)\[\]\{\}]')
        self._dot_regex = re.compile(r'\.{2,}')
        
        # VRAM Management
        self.vram_timeout = vram_timeout
        self._last_xtts_use = 0.0
        self._last_whisper_use = 0.0
        self._xtts_on_cuda = False
        self._whisper_on_cuda = False
        self._xtts_busy = False
        self._whisper_busy = False
        self._vram_lock = threading.Lock()

        try:
            # Auto-agree to Coqui TOS to prevent the server from hanging on startup
            os.environ["COQUI_TOS_AGREED"] = "1"
            # pyrefly: ignore [missing-import]
            from TTS.api import TTS
            print("[VoiceCloner] Loading Coqui XTTS v2 model (this may take a moment)...")
            self.tts_engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
            print("[VoiceCloner] XTTS v2 loaded (initially pinned to CPU to save VRAM).")
            
            # Start VRAM cleanup daemon thread
            if self.device == "cuda":
                self._cleanup_thread = threading.Thread(target=self._vram_cleanup_loop, daemon=True)
                self._cleanup_thread.start()
                
            # Model Warmup (if profile_1 is available)
            ref_path = _PROFILE_VOICE_MAP.get("profile_1")
            if ref_path and os.path.exists(ref_path):
                print("[VoiceCloner] Performing model warmup...")
                self._xtts_tts("Warmup.", ref_path)
                print("[VoiceCloner] Warmup complete.")
                
        except Exception as e:
            print(f"[VoiceCloner] WARNING: Could not load Coqui TTS: {e}")
            print("[VoiceCloner] Falling back to DSP-only pitch shifting.")

    def _vram_cleanup_loop(self):
        """Background daemon that unloads models from CUDA if they've been idle."""
        while True:
            time.sleep(2.0)
            now = time.time()
            with self._vram_lock:
                # Cleanup XTTS
                if self._xtts_on_cuda and self.tts_engine is not None and not self._xtts_busy and (now - self._last_xtts_use) > self.vram_timeout:
                    print("[VoiceCloner] Idle timeout reached. Unloading XTTS to CPU.")
                    self.tts_engine.to("cpu")
                    self._xtts_on_cuda = False
                    torch.cuda.empty_cache()
                
                # Cleanup Whisper
                if self._whisper_on_cuda and self._whisper_model is not None and not self._whisper_busy and (now - self._last_whisper_use) > self.vram_timeout:
                    print("[VoiceCloner] Idle timeout reached. Unloading Whisper to CPU.")
                    self._whisper_model.to("cpu")
                    self._whisper_on_cuda = False
                    torch.cuda.empty_cache()

    # ── Public API ────────────────────────────────────────────────────

    def clone_tts(self, text: str, profile: str = "profile_1",
                  target_voice_path: str = None, language: str = "en",
                  speed_ratio: float = 1.0, emotion: str = "default") -> np.ndarray:
        """
        Text-to-Speech with zero-shot voice cloning.

        Args:
            text:              The text to synthesise.
            profile:           Built-in profile key (used if target_voice_path is None).
            target_voice_path: Path to a WAV file of the target speaker (overrides profile).
            language:          Language code for XTTS (default: English).
            speed_ratio:       Speaking rate scaling factor (0.5 to 2.0).
            emotion:           Target emotion cue for Coqui XTTS.

        Returns:
            numpy array of synthesised audio at self.sr sample rate.
        """
        # Determine the reference voice file
        ref_path = target_voice_path or _PROFILE_VOICE_MAP.get(profile)

        # Do NOT prepend emotional tags in text as Coqui XTTS v2 will read them out.
        # Clean emotional tags from the text
        processed_text = text

        if self.tts_engine is not None and ref_path and os.path.isfile(ref_path):
            y = self._xtts_tts(processed_text, ref_path, language)
        else:
            # Fallback: use pyttsx3 + pitch shift
            y = self._fallback_tts(text, speed_ratio)

        # Apply premium, echo-free, non-robotic emotional DSP styling
        if emotion == "happy":
            print("[VoiceCloner] Applying 'happy' DSP style (speed stretch + high-frequency crispness)")
            # 1. Cheerful time stretch (1.12x)
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 1.12, n_fft=512, hop_length=128)
            # 2. Dynamic volume scaling (Gain: 1.15x)
            y = y * 1.15
            # 3. Mild pre-emphasis filter for crisp sibilants
            y = np.append(y[0], y[1:] - 0.25 * y[:-1])
            
        elif emotion == "sad":
            print("[VoiceCloner] Applying 'sad' DSP style (slow stretch + muffled low-pass filter)")
            # 1. Melancholy time stretch (0.8x)
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 0.8, n_fft=512, hop_length=128)
            # 2. Quiet volume scaling (Gain: 0.7x)
            y = y * 0.7
            # 3. De-emphasis low-pass filter to muffle voice
            y = np.append(y[0], 0.75 * y[1:] + 0.25 * y[:-1])
            
        elif emotion == "angry":
            print("[VoiceCloner] Applying 'angry' DSP style (fast stretch + saturation + high amplitude)")
            # 1. Intense time stretch (1.18x)
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 1.18, n_fft=512, hop_length=128)
            # 2. High amplitude scaling (Gain: 1.3x)
            y = y * 1.3
            # 3. Premium soft-clipping saturation to add vocal grit/aggression
            y = np.tanh(y * 1.35) / 1.35
            
        elif emotion == "whispering":
            print("[VoiceCloner] Applying 'whispering' DSP style (breath sibilance + low volume)")
            # 1. Soft time stretch (0.85x)
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 0.85, n_fft=512, hop_length=128)
            # 2. Low amplitude scaling (Gain: 0.45x)
            y = y * 0.45
            # 3. Deep pre-emphasis filter to cut bass and maximize breath sibilants
            y = np.append(y[0], y[1:] - 0.96 * y[:-1])
            
        else:
            # Default or other: apply standard speed time-stretch if requested
            if speed_ratio != 1.0 and speed_ratio > 0.0:
                print(f"[VoiceCloner] Applying DSP time-stretch with rate={speed_ratio}...")
                y = librosa.effects.time_stretch(y=y, rate=speed_ratio, n_fft=512, hop_length=128)

        return y

    def clone_voice(self, source_audio: np.ndarray, profile: str = "profile_1",
                    target_voice_path: str = None, language: str = "en",
                    speed_ratio: float = 1.0, emotion: str = "default") -> tuple[np.ndarray, str]:
        """
        Audio-to-Audio voice conversion via transcription + re-synthesis.

        Pipeline: Source Audio → Whisper (transcribe) → XTTS (re-synthesise in target voice)

        Args:
            source_audio:      numpy array of the source speech (16kHz float32).
            profile:           Built-in profile key (used if target_voice_path is None).
            target_voice_path: Path to a WAV file of the target speaker.
            language:          Language code.
            speed_ratio:       Speaking rate scaling factor (0.5 to 2.0).
            emotion:           Target emotion cue for Coqui XTTS.

        Returns:
            Tuple of (audio_array, transcript_string)
        """
        ref_path = target_voice_path or _PROFILE_VOICE_MAP.get(profile)

        if self.tts_engine is not None and ref_path and os.path.isfile(ref_path):
            # Step 1: Transcribe the source audio using Whisper
            transcript = self._transcribe(source_audio)
            if not transcript or transcript.strip() == "":
                print("[VoiceCloner] Transcription returned empty — returning pitch-shifted fallback.")
                y_shifted = self._fallback_pitch_shift(source_audio)
                if speed_ratio != 1.0 and speed_ratio > 0.0:
                    y_shifted = librosa.effects.time_stretch(y=y_shifted, rate=speed_ratio, n_fft=512, hop_length=128)
                return y_shifted, ""

            # Step 2: Re-synthesise using XTTS in the target voice (WITHOUT emotional tag text)
            y = self._xtts_tts(transcript, ref_path, language)
            
            # Apply premium, echo-free, non-robotic emotional DSP styling
            if emotion == "happy":
                print("[VoiceCloner] Applying 'happy' DSP style (speed stretch + high-frequency crispness)")
                # 1. Cheerful time stretch (1.12x)
                y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 1.12, n_fft=512, hop_length=128)
                # 2. Dynamic volume scaling (Gain: 1.15x)
                y = y * 1.15
                # 3. Mild pre-emphasis filter for crisp sibilants
                y = np.append(y[0], y[1:] - 0.25 * y[:-1])
                
            elif emotion == "sad":
                print("[VoiceCloner] Applying 'sad' DSP style (slow stretch + muffled low-pass filter)")
                # 1. Melancholy time stretch (0.8x)
                y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 0.8, n_fft=512, hop_length=128)
                # 2. Quiet volume scaling (Gain: 0.7x)
                y = y * 0.7
                # 3. De-emphasis low-pass filter to muffle voice
                y = np.append(y[0], 0.75 * y[1:] + 0.25 * y[:-1])
                
            elif emotion == "angry":
                print("[VoiceCloner] Applying 'angry' DSP style (fast stretch + saturation + high amplitude)")
                # 1. Intense time stretch (1.18x)
                y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 1.18, n_fft=512, hop_length=128)
                # 2. High amplitude scaling (Gain: 1.3x)
                y = y * 1.3
                # 3. Premium soft-clipping saturation to add vocal grit/aggression
                y = np.tanh(y * 1.35) / 1.35
                
            elif emotion == "whispering":
                print("[VoiceCloner] Applying 'whispering' DSP style (breath sibilance + low volume)")
                # 1. Soft time stretch (0.85x)
                y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 0.85, n_fft=512, hop_length=128)
                # 2. Low amplitude scaling (Gain: 0.45x)
                y = y * 0.45
                # 3. Deep pre-emphasis filter to cut bass and maximize breath sibilants
                y = np.append(y[0], y[1:] - 0.96 * y[:-1])
                
            else:
                # Default or other: apply standard speed time-stretch if requested
                if speed_ratio != 1.0 and speed_ratio > 0.0:
                    print(f"[VoiceCloner] Applying DSP time-stretch with rate={speed_ratio}...")
                    y = librosa.effects.time_stretch(y=y, rate=speed_ratio, n_fft=512, hop_length=128)
                    
            return y, transcript
        else:
            # Fallback: basic pitch shift
            y_shifted = self._fallback_pitch_shift(source_audio)
            
            # Apply premium emotional DSP styling to fallback
            if emotion == "happy":
                y_shifted = librosa.effects.time_stretch(y=y_shifted, rate=speed_ratio * 1.12, n_fft=512, hop_length=128)
                y_shifted = y_shifted * 1.15
                y_shifted = np.append(y_shifted[0], y_shifted[1:] - 0.25 * y_shifted[:-1])
            elif emotion == "sad":
                y_shifted = librosa.effects.time_stretch(y=y_shifted, rate=speed_ratio * 0.8, n_fft=512, hop_length=128)
                y_shifted = y_shifted * 0.7
                y_shifted = np.append(y_shifted[0], 0.75 * y_shifted[1:] + 0.25 * y_shifted[:-1])
            elif emotion == "angry":
                y_shifted = librosa.effects.time_stretch(y=y_shifted, rate=speed_ratio * 1.18, n_fft=512, hop_length=128)
                y_shifted = y_shifted * 1.3
                y_shifted = np.tanh(y_shifted * 1.35) / 1.35
            elif emotion == "whispering":
                y_shifted = librosa.effects.time_stretch(y=y_shifted, rate=speed_ratio * 0.85, n_fft=512, hop_length=128)
                y_shifted = y_shifted * 0.45
                y_shifted = np.append(y_shifted[0], y_shifted[1:] - 0.96 * y_shifted[:-1])
            else:
                if speed_ratio != 1.0 and speed_ratio > 0.0:
                    y_shifted = librosa.effects.time_stretch(y=y_shifted, rate=speed_ratio, n_fft=512, hop_length=128)

            return y_shifted, ""

    def pitch_shift(self, y: np.ndarray, n_steps: float = -2.0, sr: int = None) -> np.ndarray:
        """
        DSP pitch shift using librosa Phase Vocoder.
        Used for the fallback and to shift the XTTS output.
        """
        sample_rate = sr if sr is not None else 16000
        # Use high time-resolution STFT window settings (n_fft=512, hop_length=128)
        # to eliminate phase smearing, robotic metallic echoes, and watery artifacts on speech.
        return librosa.effects.pitch_shift(
            y=y,
            sr=sample_rate,
            n_steps=n_steps,
            n_fft=512,
            hop_length=128
        )

    # ── Internal: XTTS ────────────────────────────────────────────────

    def _xtts_tts(self, text: str, speaker_wav_path: str, language: str = "en") -> np.ndarray:
        """
        Uses Coqui XTTS v2 to synthesise speech from text, cloning the voice
        from the provided speaker_wav reference.

        Uses tts() (direct numpy output) instead of tts_to_file() to avoid
        the PyTorch 2.9+ torchcodec file-I/O requirement.
        """
        # Clean text: XTTS can struggle with brackets and consecutive punctuation, 
        # treating them as characters which causes noticeable delays or artifacts.
        clean_text = self._bracket_regex.sub('', text)
        clean_text = self._dot_regex.sub('.', clean_text)
        clean_text = clean_text.strip()

        try:
            with self._vram_lock:
                if self.device == "cuda" and not self._xtts_on_cuda:
                    print(f"[VoiceCloner] Loading XTTS to CUDA...")
                    self.tts_engine.to("cuda")
                    self._xtts_on_cuda = True
                self._last_xtts_use = time.time()
                self._xtts_busy = True
                
            try:
                print(f"[VoiceCloner] Running XTTS synthesis (speaker: {os.path.basename(speaker_wav_path)})...")
                # tts() returns a list of float samples at the model's native rate (24kHz)
                wav = self.tts_engine.tts(
                    text=clean_text,
                    speaker_wav=speaker_wav_path,
                    language=language,
                    split_sentences=True
                )
                y = np.array(wav, dtype=np.float32)
            finally:
                with self._vram_lock:
                    self._xtts_busy = False
                    self._last_xtts_use = time.time()
                
            print(f"[VoiceCloner] XTTS synthesis complete — {len(y)} samples at {self.sr} Hz.")
            return y
        except Exception as e:
            import traceback
            print(f"[VoiceCloner] XTTS synthesis FAILED: {e}")
            traceback.print_exc()
            return self._fallback_tts(text)

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribes audio using OpenAI Whisper (local, offline).
        Whisper model is loaded once on first call and cached.
        """
        try:
            # pyrefly: ignore [missing-import]
            import whisper

            # Lazy-load + cache the Whisper model
            if self._whisper_model is None:
                print("[VoiceCloner] Loading Whisper 'base' model (one-time)...")
                self._whisper_model = whisper.load_model("base", device="cpu")

            with self._vram_lock:
                if self.device == "cuda" and not self._whisper_on_cuda:
                    print("[VoiceCloner] Loading Whisper to CUDA...")
                    self._whisper_model.to("cuda")
                    self._whisper_on_cuda = True
                self._last_whisper_use = time.time()
                self._whisper_busy = True

            try:
                # Pass the NumPy array directly to Whisper to bypass its internal FFmpeg dependency
                # (which crashes on Windows if FFmpeg isn't installed).
                audio_fp32 = audio.astype(np.float32)
                
                # Debug audio properties
                duration = len(audio_fp32) / 16000.0
                rms = np.sqrt(np.mean(audio_fp32**2)) if len(audio_fp32) > 0 else 0.0
                print(f"[VoiceCloner] Transcribing audio: {duration:.2f}s duration, {rms:.5f} RMS volume.")

                result = self._whisper_model.transcribe(
                    audio_fp32,
                    fp16=False
                )
            finally:
                with self._vram_lock:
                    self._whisper_busy = False
                    self._last_whisper_use = time.time()
            
            transcript = result.get("text", "")
            print(f"[VoiceCloner] Transcribed: \"{transcript[:80]}...\"")
            return transcript
        except Exception as e:
            print(f"[VoiceCloner] Whisper transcription failed: {e}")
            return ""

    # ── Fallbacks ─────────────────────────────────────────────────────

    def _fallback_tts(self, text: str, speed_ratio: float = 1.0) -> np.ndarray:
        """Fallback TTS using pyttsx3."""
        tmp_path = None
        try:
            # pyrefly: ignore [missing-import]
            import pyttsx3
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            engine = pyttsx3.init()
            # Default rate is 150 words per minute. Scale it by speed_ratio.
            base_rate = 150
            engine.setProperty('rate', int(base_rate * speed_ratio))
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            # Load directly at target samplerate (e.g. 24000) to avoid downstream distortion
            y, _ = librosa.load(tmp_path, sr=self.sr)
            return y
        except Exception as e:
            print(f"[VoiceCloner] Fallback TTS also failed: {e}")
            # Return 1 second of silence
            return np.zeros(self.sr, dtype=np.float32)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _fallback_pitch_shift(self, audio: np.ndarray, n_steps: float = -2.0) -> np.ndarray:
        """Fallback audio-to-audio using simple pitch shift."""
        y_shifted = librosa.effects.pitch_shift(
            y=audio, 
            sr=16000, 
            n_steps=n_steps,
            n_fft=512,
            hop_length=128
        )
        # Resample to match clone_voice contract (self.sr)
        return librosa.resample(y_shifted, orig_sr=16000, target_sr=self.sr)

    # ── Utility ───────────────────────────────────────────────────────

    def load_audio(self, file_path: str) -> np.ndarray:
        """Loads audio and resamples to the target system rate."""
        y, _ = librosa.load(file_path, sr=self.sr)
        return y

    def get_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Generates a Mel-spectrogram.
        Math: S = Mel(STFT(y)^2)
        """
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=2048,
            hop_length=512, n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db

    def save_output(self, y: np.ndarray, output_path: str):
        """Saves the processed waveform to a file."""
        sf.write(output_path, y, self.sr)