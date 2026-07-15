"""
voice_cloning.py - Voice Cloning Module using F5-TTS
Performs high-quality zero-shot voice cloning for both TTS and audio-to-audio.
Supports preset voice profiles and user-uploaded target voice samples.
"""

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

# -- Constants --------------------------------------------------------
_MODEL_DIR   = os.path.join(os.path.dirname(__file__))
_TEMPLATES_DIR = os.path.join(_MODEL_DIR, "templates")

# Default voice reference files for built-in profiles.
# Place short WAV clips (3-10 seconds of clean speech) in the templates folder.
_PROFILE_VOICE_MAP = {
    "profile_1": os.path.join(_TEMPLATES_DIR, "profile_1_voice.wav"),
    "profile_2": os.path.join(_TEMPLATES_DIR, "profile_2_voice.wav"),
}


class VoiceCloner:
    def __init__(self, sample_rate: int = 24000, vram_timeout: float = 600.0):
        """
        Initialises the F5-TTS model for zero-shot voice cloning.
        Falls back to basic DSP pitch shifting if F5-TTS is unavailable.

        F5-TTS uses a flow-matching diffusion architecture - significantly
        faster inference than autoregressive models like XTTS v2, with
        comparable or better naturalness.
        """
        self.sr            = sample_rate
        self.tts_engine    = None           # F5TTS instance
        self._whisper_model = None          # lazy-loaded Whisper model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cache: maps original speaker_wav_path -> preprocessed temp WAV path
        # Preprocessing trims to <=12 s, resamples to 24 kHz, and normalises.
        # F5-TTS quality degrades significantly with clips longer than ~15 s.
        # Results are kept in memory only - no _f5ready.wav files written to disk.
        self._preprocessed_ref_cache: dict = {}
        self._preprocessed_ref_tmpfiles: list = []  # track temp files for cleanup

        # Cache: maps preprocessed WAV path -> transcription string.
        # F5-TTS needs the exact text spoken in the reference audio.
        self._ref_text_cache: dict = {}

        # Pre-compile regex for text cleaning
        self._bracket_regex = re.compile(r'[\(\)\[\]\{\}]')
        self._dot_regex     = re.compile(r'\.{2,}')

        # VRAM management
        self.vram_timeout       = vram_timeout
        self._last_f5_use       = 0.0
        self._last_whisper_use  = 0.0
        self._f5_on_cuda        = False
        self._whisper_on_cuda   = False
        self._f5_busy           = False
        self._whisper_busy      = False
        self._vram_lock         = threading.Lock()

        try:
            # pyrefly: ignore [missing-import]
            from f5_tts.api import F5TTS
            print("[VoiceCloner] Loading F5-TTS model (this may take a moment)...")
            # Load on CPU first - avoids a CUDA-context crash that can occur
            # when loading directly to CUDA before the runtime is fully warmed up.
            self.tts_engine = F5TTS(device="cpu")
            if self.device == "cuda":
                print("[VoiceCloner] Moving F5-TTS to CUDA (float16)...")
                self.tts_engine.ema_model.half().to("cuda")  # float16: 2x faster ODE steps
                self.tts_engine.vocoder.to("cuda")           # vocoder stays float32; F5-TTS casts mel before decode
                self.tts_engine.device = "cuda"              # must match model placement
                self._f5_on_cuda = True
                try:
                    actual_dev = next(self.tts_engine.ema_model.parameters()).device
                    print(f"[VoiceCloner] F5-TTS loaded on CUDA. Tensors: {actual_dev}")
                except Exception:
                    print("[VoiceCloner] F5-TTS moved to CUDA.")
                # Start idle VRAM cleanup daemon
                self._cleanup_thread = threading.Thread(
                    target=self._vram_cleanup_loop, daemon=True
                )
                self._cleanup_thread.start()
            else:
                print("[VoiceCloner] F5-TTS loaded on CPU (no GPU detected).")

            # Warmup pass to JIT-compile the model before first real request
            ref_path = _PROFILE_VOICE_MAP.get("profile_1")
            if ref_path and os.path.exists(ref_path):
                print("[VoiceCloner] Performing model warmup...")
                self._f5_tts("Warmup.", ref_path)
                print("[VoiceCloner] Warmup complete.")

        except Exception as e:
            print(f"[VoiceCloner] WARNING: Could not load F5-TTS: {e}")
            print("[VoiceCloner] Falling back to DSP-only pitch shifting.")

    # -- VRAM Cleanup Daemon -------------------------------------------

    def _vram_cleanup_loop(self):
        """Background daemon that frees CUDA memory when models are idle."""
        while True:
            time.sleep(2.0)
            now = time.time()
            with self._vram_lock:
                # F5-TTS idle cleanup - just flush the CUDA cache; the model
                # itself stays in GPU memory (it is much smaller than XTTS v2).
                if (self._f5_on_cuda
                        and self.tts_engine is not None
                        and not self._f5_busy
                        and (now - self._last_f5_use) > self.vram_timeout):
                    print("[VoiceCloner] F5-TTS idle timeout - flushing CUDA cache.")
                    torch.cuda.empty_cache()

                # Whisper idle cleanup
                if (self._whisper_on_cuda
                        and self._whisper_model is not None
                        and not self._whisper_busy
                        and (now - self._last_whisper_use) > self.vram_timeout):
                    print("[VoiceCloner] Whisper idle timeout - unloading to CPU.")
                    self._whisper_model.to("cpu")
                    self._whisper_on_cuda = False
                    torch.cuda.empty_cache()

    # -- Public API ----------------------------------------------------

    def clone_tts(self, text: str, profile: str = "profile_1",
                  target_voice_path: str = None, language: str = "en",
                  speed_ratio: float = 1.0, emotion: str = "default") -> np.ndarray:
        """
        Text-to-Speech with zero-shot voice cloning via F5-TTS.

        Args:
            text:              The text to synthesise.
            profile:           Built-in profile key (used if target_voice_path is None).
            target_voice_path: Path to a WAV file of the target speaker (overrides profile).
            language:          Language hint (informational; F5-TTS is multilingual).
            speed_ratio:       Speaking rate scaling factor (0.5 to 2.0).
            emotion:           Target emotion cue - applied via DSP post-processing.

        Returns:
            numpy array of synthesised audio at self.sr sample rate.
        """
        ref_path = target_voice_path or _PROFILE_VOICE_MAP.get(profile)

        if self.tts_engine is not None and ref_path and os.path.isfile(ref_path):
            # Pass speed natively to F5-TTS: generates fewer mel frames -> faster
            # synthesis AND cleaner result than post-processing time-stretch.
            y = self._f5_tts(text, ref_path, speed=speed_ratio)
        else:
            y = self._fallback_tts(text, speed_ratio)

        # Apply DSP emotional styling (speed already handled natively above)
        y = self._apply_emotion(y, emotion, speed_ratio=1.0)
        return y

    def clone_voice(self, source_audio: np.ndarray, profile: str = "profile_1",
                    target_voice_path: str = None, language: str = "en",
                    speed_ratio: float = 1.0, emotion: str = "default") -> tuple:
        """
        Audio-to-Audio voice conversion via transcription + re-synthesis.

        Pipeline: Source Audio -> Whisper (transcribe) -> F5-TTS (re-synthesise in target voice)

        Args:
            source_audio:      numpy array of the source speech (16 kHz float32).
            profile:           Built-in profile key (used if target_voice_path is None).
            target_voice_path: Path to a WAV file of the target speaker.
            language:          Language hint.
            speed_ratio:       Speaking rate scaling factor (0.5 to 2.0).
            emotion:           Target emotion cue.

        Returns:
            Tuple of (audio_array, transcript_string)
        """
        ref_path = target_voice_path or _PROFILE_VOICE_MAP.get(profile)

        if self.tts_engine is not None and ref_path and os.path.isfile(ref_path):
            # Step 1: transcribe the source audio with Whisper
            transcript = self._transcribe(source_audio)
            if not transcript or transcript.strip() == "":
                print("[VoiceCloner] Transcription empty - returning pitch-shifted fallback.")
                y_shifted = self._fallback_pitch_shift(source_audio)
                y_shifted = self._apply_emotion(y_shifted, emotion, speed_ratio)
                return y_shifted, ""

            # Step 2: re-synthesise in the target voice with F5-TTS
            y = self._f5_tts(transcript, ref_path, speed=speed_ratio)
            y = self._apply_emotion(y, emotion, speed_ratio=1.0)
            return y, transcript
        else:
            # Fallback: basic pitch shift
            y_shifted = self._fallback_pitch_shift(source_audio)
            y_shifted = self._apply_emotion(y_shifted, emotion, speed_ratio)
            return y_shifted, ""

    def pitch_shift(self, y: np.ndarray, n_steps: float = -2.0, sr: int = None) -> np.ndarray:
        """
        DSP pitch shift using librosa Phase Vocoder.
        """
        sample_rate = sr if sr is not None else 16000
        return librosa.effects.pitch_shift(
            y=y,
            sr=sample_rate,
            n_steps=n_steps,
            n_fft=512,
            hop_length=128
        )

    # -- Internal: Emotion DSP -----------------------------------------

    def _apply_emotion(self, y: np.ndarray, emotion: str, speed_ratio: float = 1.0) -> np.ndarray:
        """Applies premium, echo-free DSP emotional styling to the audio."""
        if emotion == "happy":
            print("[VoiceCloner] Applying 'happy' DSP style (speed stretch + high-frequency crispness)")
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 1.12, n_fft=512, hop_length=128)
            y = y * 1.15
            y = np.append(y[0], y[1:] - 0.25 * y[:-1])

        elif emotion == "sad":
            print("[VoiceCloner] Applying 'sad' DSP style (slow stretch + muffled low-pass filter)")
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 0.8, n_fft=512, hop_length=128)
            y = y * 0.7
            y = np.append(y[0], 0.75 * y[1:] + 0.25 * y[:-1])

        elif emotion == "angry":
            print("[VoiceCloner] Applying 'angry' DSP style (fast stretch + saturation + high amplitude)")
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 1.18, n_fft=512, hop_length=128)
            y = y * 1.3
            y = np.tanh(y * 1.35) / 1.35

        elif emotion == "whispering":
            print("[VoiceCloner] Applying 'whispering' DSP style (breath sibilance + low volume)")
            y = librosa.effects.time_stretch(y=y, rate=speed_ratio * 0.85, n_fft=512, hop_length=128)
            y = y * 0.45
            y = np.append(y[0], y[1:] - 0.96 * y[:-1])

        else:
            # Default: apply speed time-stretch only if requested
            if speed_ratio != 1.0 and speed_ratio > 0.0:
                print(f"[VoiceCloner] Applying DSP time-stretch with rate={speed_ratio}...")
                y = librosa.effects.time_stretch(y=y, rate=speed_ratio, n_fft=512, hop_length=128)

        return y

    # -- Internal: Reference Audio Preprocessing ---------------------

    # F5-TTS quality sweet spot: 3-15 seconds of clean, normalised speech.
    # Clips longer than this confuse the speaker conditioning and produce
    # robotic or distorted output.
    _REF_MAX_SECONDS = 12.0
    _REF_SAMPLE_RATE = 24000
    _REF_TARGET_RMS  = 0.10

    def _preprocess_ref_audio(self, speaker_wav_path: str) -> str:
        """
        Prepares a reference WAV for F5-TTS entirely in memory:
          1. Loads the original file (any sample rate).
          2. Trims to at most _REF_MAX_SECONDS (12 s) - the F5-TTS sweet spot.
          3. Resamples to _REF_SAMPLE_RATE (24 kHz).
          4. Normalises RMS amplitude to _REF_TARGET_RMS (0.10).
          5. Writes to a temp file (no permanent _f5ready.wav on disk).

        The result is cached in memory so repeated calls with the same source
        file (e.g. a profile voice) are instant.
        Returns the path to the ready-to-use temp WAV.
        """
        if speaker_wav_path in self._preprocessed_ref_cache:
            cached = self._preprocessed_ref_cache[speaker_wav_path]
            if os.path.isfile(cached):
                return cached

        t0 = time.time()
        print(f"[VoiceCloner] Preprocessing reference audio: "
              f"{os.path.basename(speaker_wav_path)}")

        # Load at native rate
        y_orig, sr_orig = librosa.load(speaker_wav_path, sr=None, mono=True)
        orig_dur = len(y_orig) / sr_orig

        # Trim to 12 s
        max_samples = int(self._REF_MAX_SECONDS * sr_orig)
        y_trimmed = y_orig[:max_samples] if len(y_orig) > max_samples else y_orig
        trim_dur = len(y_trimmed) / sr_orig

        # Resample to 24 kHz
        if sr_orig != self._REF_SAMPLE_RATE:
            y_resampled = librosa.resample(
                y_trimmed, orig_sr=sr_orig, target_sr=self._REF_SAMPLE_RATE
            )
        else:
            y_resampled = y_trimmed

        # Normalise RMS
        rms = np.sqrt(np.mean(y_resampled ** 2))
        y_norm = y_resampled * (self._REF_TARGET_RMS / rms) if rms > 1e-6 else y_resampled

        # Write to a temp file (lives for the duration of the server process)
        tmp = tempfile.NamedTemporaryFile(suffix='_f5ref.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()
        sf.write(tmp_path, y_norm, self._REF_SAMPLE_RATE, subtype='PCM_16')

        self._preprocessed_ref_cache[speaker_wav_path] = tmp_path
        self._preprocessed_ref_tmpfiles.append(tmp_path)

        elapsed = (time.time() - t0) * 1000
        print(f"[VoiceCloner]   {orig_dur:.1f}s -> {trim_dur:.1f}s | "
              f"{sr_orig} Hz -> {self._REF_SAMPLE_RATE} Hz | "
              f"RMS {rms:.3f} -> {self._REF_TARGET_RMS} | "
              f"done in {elapsed:.0f} ms")
        return tmp_path

    # -- Internal: F5-TTS ---------------------------------------------

    # Maximum characters per chunk sent to F5-TTS.
    # Larger chunks = fewer infer() calls = less per-call overhead.
    # F5-TTS handles up to ~500 chars cleanly before quality drops.
    _F5_CHUNK_SIZE = 400

    @staticmethod
    def _crossfade_concat(parts: list, sr: int, fade_ms: int = 25) -> np.ndarray:
        """Concatenate audio parts with a short linear cross-fade to avoid clicks."""
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0]
        fade = max(1, int(sr * fade_ms / 1000))
        result = parts[0].copy()
        for nxt in parts[1:]:
            if len(result) < fade or len(nxt) < fade:
                result = np.concatenate([result, nxt])
                continue
            fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            fade_in  = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            result[-fade:] *= fade_out
            nxt_copy = nxt.copy()
            nxt_copy[:fade] *= fade_in
            result = np.concatenate([result, nxt_copy])
        return result

    def _get_ref_text(self, preprocessed_wav_path: str) -> str:
        """
        Returns the transcription of the (already-preprocessed) reference WAV.
        The full 12-second clip is transcribed so that F5-TTS receives an
        accurate speaking-rate estimate.  A short or wrong ref_text causes
        F5-TTS to wildly overestimate per-batch duration -> stretched, alien,
        distorted output.  The result is cached after the first call.
        """
        if preprocessed_wav_path in self._ref_text_cache:
            return self._ref_text_cache[preprocessed_wav_path]

        print(f"[VoiceCloner] Transcribing reference clip: "
              f"{os.path.basename(preprocessed_wav_path)} (full clip, one-time)...")
        try:
            t0 = time.time()
            # Load the full preprocessed clip (<=12 s) so Whisper has enough
            # context to avoid hallucinations and produce an accurate transcript.
            audio, _ = librosa.load(
                preprocessed_wav_path, sr=16000
            )
            ref_text = self._transcribe(audio)
            if not ref_text or not ref_text.strip():
                ref_text = ""
            self._ref_text_cache[preprocessed_wav_path] = ref_text
            elapsed = (time.time() - t0) * 1000
            print(f"[VoiceCloner] ref_text ({elapsed:.0f} ms): \"{ref_text[:100]}\"")
            return ref_text
        except Exception as e:
            print(f"[VoiceCloner] Could not transcribe reference wav: {e}")
            self._ref_text_cache[preprocessed_wav_path] = ""
            return ""

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int) -> list:
        """
        Splits text into chunks of at most max_chars, breaking only at sentence
        boundaries (. ! ?) to avoid mid-sentence cuts.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current = [], ""
        for sentence in sentences:
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    chunks.append(current)
                # Single sentence longer than max_chars - split at commas
                if len(sentence) > max_chars:
                    parts = sentence.split(',')
                    current = ""
                    for part in parts:
                        part = part.strip()
                        if len(current) + len(part) + 2 <= max_chars:
                            current = (current + ", " + part).strip(", ")
                        else:
                            if current:
                                chunks.append(current)
                            current = part
                    if current:
                        chunks.append(current)
                    current = ""
                else:
                    current = sentence
        if current:
            chunks.append(current)
        return chunks if chunks else [text]

    def _f5_tts(self, text: str, speaker_wav_path: str, speed: float = 1.0) -> np.ndarray:
        """
        Uses F5-TTS to synthesise speech from text, cloning the voice from
        the provided speaker_wav reference.

        Pipeline:
          1. Preprocess reference audio (trim to <=12 s, resample, normalise).
          2. Transcribe the preprocessed clip with Whisper (cached).
          3. Offload Whisper from CUDA to free VRAM for F5-TTS.
          4. Single infer() call - F5-TTS handles sentence batching internally,
             encoding the reference audio only once. Far faster than calling
             infer() per chunk which re-encodes the reference every time.
        """
        # Clean text
        clean_text = self._bracket_regex.sub('', text)
        clean_text = self._dot_regex.sub('.', clean_text).strip()
        if not clean_text:
            return np.zeros(0, dtype=np.float32)

        # Step 1: preprocess reference (trim, resample, normalise)
        ref_path = self._preprocess_ref_audio(speaker_wav_path)

        # Step 2: get reference transcript for duration calculation (cached).
        # IMPORTANT: we use our own Whisper on the full 12s clip so the
        # speaking-rate ratio is accurate.  F5-TTS's internal transcription
        # downloads whisper-large-v3-turbo (1.62 GB) - too heavy.
        ref_text = self._get_ref_text(ref_path)

        # Step 3: offload OUR Whisper so F5-TTS has headroom in VRAM.
        with self._vram_lock:
            if self._whisper_on_cuda and self._whisper_model is not None and not self._whisper_busy:
                print("[VoiceCloner] Offloading Whisper to CPU before F5-TTS...")
                self._whisper_model.to("cpu")
                self._whisper_on_cuda = False
                torch.cuda.empty_cache()

        with self._vram_lock:
            self._last_f5_use = time.time()
            self._f5_busy = True

        try:
            # Split into <=250-char segments so F5-TTS's internal ThreadPoolExecutor
            # never submits more than ~4 batches concurrently.  Submitting all
            # batches of a long text at once exhausts VRAM on 4 GB GPUs and crashes.
            segments = self._split_into_chunks(clean_text, max_chars=250)
            n_segs = len(segments)
            print(f"[VoiceCloner] F5-TTS synthesising {len(clean_text)} chars "
                  f"in {n_segs} segment(s) (ref='{os.path.basename(ref_path)}')...")
            if ref_text:
                print(f"[VoiceCloner] ref_text ({len(ref_text)} chars): \"{ref_text[:80]}{'...' if len(ref_text)>80 else ''}\"")

            t0 = time.time()
            audio_parts: list = []
            out_sr = self.sr

            for i, segment in enumerate(segments, 1):
                if n_segs > 1:
                    print(f"[VoiceCloner]   segment {i}/{n_segs} ({len(segment)} chars)")
                seg_wav, seg_sr, _ = self.tts_engine.infer(
                    ref_file=ref_path,
                    ref_text=ref_text,
                    gen_text=segment,
                    nfe_step=16,
                    cfg_strength=2.0,
                    cross_fade_duration=0.15,
                    remove_silence=True,
                    speed=speed,
                    show_info=print,
                )
                audio_parts.append(np.array(seg_wav, dtype=np.float32))
                out_sr = seg_sr
                if self._f5_on_cuda:
                    torch.cuda.empty_cache()   # free intermediate VRAM between segments

            # Cross-fade between segments to avoid clicks at join points
            y = self._crossfade_concat(audio_parts, out_sr or self.sr, fade_ms=25)
            elapsed = time.time() - t0
            dur = len(y) / (out_sr or self.sr)
            print(f"[VoiceCloner] Done - {dur:.2f}s audio in {elapsed:.2f}s "
                  f"(RTF {elapsed/max(dur,0.01):.2f}x)")
            return y

        except Exception as e:
            import traceback
            print(f"[VoiceCloner] F5-TTS synthesis FAILED: {e}")
            traceback.print_exc()
            return self._fallback_tts(text)

        finally:
            with self._vram_lock:
                self._f5_busy = False
                self._last_f5_use = time.time()

    # -- Internal: Whisper Transcription ------------------------------

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribes audio using OpenAI Whisper (local, offline).
        Whisper model is loaded once on first call and cached.
        """
        try:
            # pyrefly: ignore [missing-import]
            import whisper

            if self._whisper_model is None:
                print("[VoiceCloner] Loading Whisper 'base' model (one-time, better accuracy than tiny)...")
                self._whisper_model = whisper.load_model("base", device="cpu")

            # Whisper stays on CPU. On a 4 GB card the GPU is already full
            # (F5-TTS + buffalo_l + inswapper + CodeFormer), and moving Whisper
            # to CUDA tips it over and hard-crashes the process. 'base' on CPU
            # transcribes a ~12 s clip in a couple of seconds and runs rarely.
            with self._vram_lock:
                self._last_whisper_use = time.time()
                self._whisper_busy = True

            try:
                audio_fp32 = audio.astype(np.float32)
                duration   = len(audio_fp32) / 16000.0
                rms        = np.sqrt(np.mean(audio_fp32 ** 2)) if len(audio_fp32) > 0 else 0.0
                print(f"[VoiceCloner] Transcribing audio (CPU): {duration:.2f}s, RMS={rms:.5f}")
                result = self._whisper_model.transcribe(audio_fp32, fp16=False)
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

    # -- Fallbacks -----------------------------------------------------

    def _fallback_tts(self, text: str, speed_ratio: float = 1.0) -> np.ndarray:
        """Fallback TTS using pyttsx3 (offline, no voice cloning)."""
        tmp_path = None
        try:
            # pyrefly: ignore [missing-import]
            import pyttsx3
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            engine = pyttsx3.init()
            engine.setProperty('rate', int(150 * speed_ratio))
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            y, _ = librosa.load(tmp_path, sr=self.sr)
            return y
        except Exception as e:
            print(f"[VoiceCloner] Fallback TTS also failed: {e}")
            return np.zeros(self.sr, dtype=np.float32)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _fallback_pitch_shift(self, audio: np.ndarray, n_steps: float = -2.0) -> np.ndarray:
        """Fallback audio-to-audio using simple DSP pitch shift."""
        y_shifted = librosa.effects.pitch_shift(
            y=audio,
            sr=16000,
            n_steps=n_steps,
            n_fft=512,
            hop_length=128
        )
        return librosa.resample(y_shifted, orig_sr=16000, target_sr=self.sr)

    # -- Utility -------------------------------------------------------

    def load_audio(self, file_path: str) -> np.ndarray:
        """Loads audio and resamples to the target system rate."""
        y, _ = librosa.load(file_path, sr=self.sr)
        return y

    def get_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Generates a Mel-spectrogram.
        Math: S = Mel(STFT(y)^2)
        """
        S    = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=2048, hop_length=512, n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
