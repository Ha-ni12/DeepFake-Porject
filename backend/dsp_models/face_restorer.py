"""
face_restorer.py — GFPGAN ONNX face restoration wrapper.

Loads `GFPGANv1.4.onnx` from this directory. If the file is missing the
restorer will attempt a one-time auto-download (~333 MB) on first server
start. If download or load fails, the restorer becomes a no-op so the
swap pipeline still works.

Model I/O contract (GFPGANv1.4.onnx):
  input  : "input"  shape [1, 3, 512, 512] float32, BGR→RGB, normalized to [-1, 1]
  output : float32 image, same shape, [-1, 1]

Public API:
  GFPGANRestorer.available -> bool
  GFPGANRestorer.restore(face_bgr_uint8) -> restored uint8 face (same shape as input)
"""

import os
import sys
import cv2
import numpy as np
import urllib.request

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "GFPGANv1.4.onnx")
# Mirrors are tried in order. First successful download wins.
# (xuanandsix's old GitHub release 404'd, so HF mirrors come first.)
_MODEL_URLS = [
    "https://huggingface.co/Meeperomi/GFPGANv1.4-onnx/resolve/main/GFPGANv1.4.onnx",
    "https://huggingface.co/neurobytemind/GFPGANv1.4.onnx/resolve/main/GFPGANv1.4.onnx",
]
_INPUT_SIZE = 512


# ── Cached ONNX provider resolution ──────────────────────────────────
# Probe CUDA / DirectML once at startup so model loads don't spam EP errors
# when accelerators are listed as available but fail at runtime (common on
# machines with onnxruntime-gpu installed but no NVIDIA driver, e.g. AMD GPUs).
_PROVIDERS_CACHE = None


def _provider_works(name: str) -> bool:
    """
    Try to create a tiny session with ONLY the given provider. Strict mode:
    we disable CPU EP fallback so a CUDA failure raises instead of silently
    using CPU. Native stderr is redirected to /dev/null during the probe to
    suppress onnxruntime's noisy EP error stack trace.
    """
    try:
        import onnxruntime as ort
        from onnx import helper, TensorProto, save_model
        import tempfile

        so = ort.SessionOptions()
        so.log_severity_level = 4  # fatal only
        try:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        except Exception:
            pass

        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "probe", [inp], [out])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp.close()
        save_model(model, tmp.name)

        # Redirect native (C-level) stdout AND stderr fds to devnull during
        # the probe — onnxruntime's Python wrapper prints "EP Error ..."
        # retry messages to stdout, and the C++ layer writes to stderr.
        saved_out_fd = None
        saved_err_fd = None
        devnull_fd = None
        saved_py_stdout = sys.stdout
        saved_py_stderr = sys.stderr
        try:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                saved_out_fd = os.dup(1)
                saved_err_fd = os.dup(2)
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                # Also swap the Python-level streams so any `print()` in
                # onnxruntime's wrapper goes to a black hole rather than to
                # the buffered original sys.stdout (which would flush back
                # to the real fd 1 after we restore).
                import io as _io
                sys.stdout = _io.StringIO()
                sys.stderr = _io.StringIO()
            except Exception:
                saved_out_fd = None
                saved_err_fd = None  # best-effort only

            try:
                sess = ort.InferenceSession(tmp.name, sess_options=so, providers=[name])
                return name in sess.get_providers()
            except Exception:
                return False
        finally:
            # Restore Python-level streams first
            sys.stdout = saved_py_stdout
            sys.stderr = saved_py_stderr
            # Restore stdout / stderr fds
            if saved_out_fd is not None:
                try:
                    os.dup2(saved_out_fd, 1)
                    os.close(saved_out_fd)
                except Exception:
                    pass
            if saved_err_fd is not None:
                try:
                    os.dup2(saved_err_fd, 2)
                    os.close(saved_err_fd)
                except Exception:
                    pass
            if devnull_fd is not None:
                try:
                    os.close(devnull_fd)
                except Exception:
                    pass
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    except Exception:
        return False


def resolve_providers():
    """Returns ONNX Runtime execution providers in priority order, cached."""
    global _PROVIDERS_CACHE
    if _PROVIDERS_CACHE is not None:
        return _PROVIDERS_CACHE

    import onnxruntime as ort
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available and _provider_works("CUDAExecutionProvider"):
        # Restrict ONNX to 1.5 GB VRAM and prevent aggressive arena growth
        cuda_options = {
            "device_id": 0,
            "gpu_mem_limit": int(1.5 * 1024 * 1024 * 1024),  # 1.5 GB
            "arena_extend_strategy": "kSameAsRequested",
        }
        chosen = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        print("[ORT] Using CUDA execution provider (NVIDIA GPU).")
    elif "DmlExecutionProvider" in available and _provider_works("DmlExecutionProvider"):
        chosen = ["DmlExecutionProvider", "CPUExecutionProvider"]
        print("[ORT] Using DirectML execution provider (AMD/Intel/NVIDIA GPU).")
    else:
        chosen = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            print("[ORT] CUDA listed but unusable (driver/DLL missing). "
                  "Using CPU. For AMD GPUs install: pip install onnxruntime-directml")
        else:
            print("[ORT] No GPU provider available. Using CPU.")

    _PROVIDERS_CACHE = chosen
    return chosen


def _download_model() -> bool:
    """Downloads GFPGANv1.4.onnx with a progress bar. Returns True on success."""
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, 100.0 * downloaded / total_size)
            mb = downloaded / (1024 * 1024)
            tot = total_size / (1024 * 1024)
            sys.stdout.write(f"\r[GFPGAN]   {pct:5.1f}%  {mb:6.1f} / {tot:.1f} MB")
            sys.stdout.flush()

    for url in _MODEL_URLS:
        try:
            print(f"[GFPGAN] Downloading model (~333 MB) from {url}")
            print(f"[GFPGAN]   -> {_MODEL_PATH}")
            urllib.request.urlretrieve(url, _MODEL_PATH, _progress)
            print()  # newline after progress bar
            print("[GFPGAN] Download complete.")
            return True
        except Exception as e:
            print(f"\n[GFPGAN] Mirror failed: {e}")
            if os.path.isfile(_MODEL_PATH):
                try:
                    os.remove(_MODEL_PATH)
                except OSError:
                    pass
            # Try next mirror
            continue

    print("[GFPGAN] All mirrors exhausted.")
    return False


class GFPGANRestorer:
    def __init__(self):
        self.available = False
        self.session = None
        self.input_name = None

        # Auto-download on first run if the model is missing.
        # Set DISABLE_GFPGAN_AUTODOWNLOAD=1 in the environment to skip.
        if not os.path.isfile(_MODEL_PATH):
            if os.environ.get("DISABLE_GFPGAN_AUTODOWNLOAD") == "1":
                print(
                    f"[GFPGAN] Model not found at {_MODEL_PATH} and "
                    "auto-download disabled. Face restoration disabled."
                )
                return
            if not _download_model():
                print("[GFPGAN] Continuing without face restoration.")
                return

        try:
            import onnxruntime as ort
            providers = resolve_providers()
            self.session = ort.InferenceSession(_MODEL_PATH, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.available = True
            print(f"[GFPGAN] Loaded GFPGANv1.4.onnx (providers={self.session.get_providers()}).")
        except Exception as e:
            print(f"[GFPGAN] Failed to load model: {e}. Restoration disabled.")
            self.session = None
            self.available = False

    def restore(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Restores a tightly-cropped face image. Input/output are the same size
        (we resize internally to 512x512 for the model and back).
        """
        if not self.available or face_bgr is None or face_bgr.size == 0:
            return face_bgr

        try:
            h, w = face_bgr.shape[:2]
            # BGR -> RGB, resize to 512x512, normalize to [-1, 1], NCHW
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            x = resized.astype(np.float32) / 255.0
            x = (x - 0.5) / 0.5
            x = np.transpose(x, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

            out = self.session.run(None, {self.input_name: x})[0]
            out = out[0]  # CHW
            out = np.transpose(out, (1, 2, 0))
            out = (out * 0.5 + 0.5) * 255.0
            out = np.clip(out, 0, 255).astype(np.uint8)

            # RGB -> BGR, resize back to original face crop size
            restored_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            if (h, w) != (_INPUT_SIZE, _INPUT_SIZE):
                restored_bgr = cv2.resize(restored_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
            return restored_bgr
        except Exception as e:
            print(f"[GFPGAN] Restore failed, returning original: {e}")
            return face_bgr
