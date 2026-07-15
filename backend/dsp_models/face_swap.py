"""
face_swap.py — Face Swapping Module using InsightFace
Performs high-quality one-shot face swapping.

Swap model (in order of preference):
  • blendswap_256.onnx (256×256) — higher-resolution output, better identity
    transfer than the 128×128 bottleneck.  Download via download_models.py.
  • inswapper_128.onnx (128×128) — fallback when blendswap_256 is absent.

Face restorer (in order of preference):
  • CodeFormer  (codeformer.onnx)   — better eye/teeth/texture fidelity.
  • GFPGANv1.4  (GFPGANv1.4.onnx)  — fallback when codeformer is absent.

Enhanced post-processing pipeline:
  1. Face alignment    — similarity transform to a normalised template.
  2. Anatomical mask   — builds precise per-region masks from 106 landmarks
                         (forehead, L/R cheek, nose, chin, jawline).
  3. Per-region colour — corrects colour in LAB space independently for each
     correction          facial zone so lighting gradients are preserved.
  4. Laplacian pyramid — multi-band frequency-domain blending for seamless
     blending            transitions (replaces simple alpha blending).
  5. Seamless clone    — final Poisson-equation pass for mathematically
                         perfect boundary integration.
  6. Unsharp mask      — recovers fine detail lost during smoothing.
"""

import os
import cv2
import warnings
import traceback
import numpy as np

# Silence the harmless FutureWarning from insightface's internal face_align.py
# (scikit-image deprecated `estimate` in favour of `from_estimate`)
warnings.filterwarnings("ignore", message=".*estimate.*is deprecated.*",
                        category=FutureWarning)

# ── InsightFace imports ──────────────────────────────────────────────
import insightface
from insightface.app import FaceAnalysis

# Optional GFPGAN / CodeFormer restorer (no-op if model file is missing).
# `resolve_providers` probes CUDA / DirectML once and caches the result
# so InsightFace model loads don't spam EP fallback errors.
from backend.dsp_models.face_restorer import GFPGANRestorer, CodeFormerRestorer, resolve_providers, _free_torch_vram

# Path constants
_MODEL_DIR = os.path.join(os.path.dirname(__file__))
_SWAPPER_MODEL      = os.path.join(_MODEL_DIR, "inswapper_128.onnx")
_BLENDSWAP_MODEL    = os.path.join(_MODEL_DIR, "blendswap_256.onnx")
_SIMSWAP_MODEL      = os.path.join(_MODEL_DIR, "simswap_256.onnx")
_CROSSFACE_SIMSWAP  = os.path.join(_MODEL_DIR, "crossface_simswap.onnx")
_TEMPLATES_DIR      = os.path.join(_MODEL_DIR, "templates")

# ── 5-point face alignment templates ────────────────────────────────
# Normalised coordinates in [0, 1]; multiply by target crop size to get pixels.
# arcface_112_v2: used for identity-extraction crops (blendswap source / SimSwap crossface)
_ARCFACE_112_V2 = np.array([
    [0.34191607, 0.46157411],
    [0.65653393, 0.45983393],
    [0.50022500, 0.64050536],
    [0.37097589, 0.82469196],
    [0.63151696, 0.82325089],
], dtype=np.float32)

# arcface_112_v1: used for SimSwap target canvas (256×256 crop of the face to be modified)
_ARCFACE_112_V1 = np.array([
    [0.35473214, 0.45658929],
    [0.64526786, 0.45658929],
    [0.50000000, 0.61154464],
    [0.37913393, 0.77687500],
    [0.62086607, 0.77687500],
], dtype=np.float32)

# ffhq_512: used for full-face alignment (blendswap target / CodeFormer input)
_FFHQ_512 = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32)

# ── 106-landmark index groups ────────────────────────────────────────
# InsightFace landmark_2d_106 topology (0-indexed):
#   0-32   : jawline contour (right ear → chin → left ear)
#   33-37  : right eyebrow
#   38-42  : left eyebrow
#   43-46  : nose bridge
#   47-54  : nose tip / nostrils
#   55-72  : outer eye contours
#   73-78  : inner eye details
#   79-87  : outer lip
#   88-95  : inner lip
#   96-100 : right eye iris
#   101-105: left eye iris

_FACE_ZONES_106 = {
    "forehead":    {"top_brow": list(range(33, 43)), "jaw_top": [0, 32]},
    "left_cheek":  {"pts": [0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 34, 35, 36, 37, 55, 56, 57]},
    "right_cheek": {"pts": [24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 64, 65, 66]},
    "nose":        {"pts": list(range(43, 55))},
    "chin":        {"pts": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]},
    "mouth":       {"pts": list(range(79, 96))},
}


class FaceSwapper:
    def __init__(self):
        """
        Initialises the InsightFace face analyser and the inswapper model.
        The face analyser handles detection + landmark extraction.
        The swapper model performs identity transfer.
        """
        import onnxruntime as _ort  # local import to avoid circular at top-level

        # ── Face analyser (buffalo_l) — detection + landmarks + ArcFace embed.
        self.app = FaceAnalysis(name="buffalo_l", providers=self._get_providers())
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # ── inswapper_128 (PRIMARY) — strongest identity transfer available.
        #    This is the model roop / facefusion use; combined with CodeFormer
        #    restoration it gives the cleanest "looks like the target" result,
        #    and its 128×128 bottleneck uses far less VRAM than SimSwap 256.
        self.swapper = None
        if os.path.isfile(_SWAPPER_MODEL):
            try:
                try:
                    self.swapper = insightface.model_zoo.get_model(
                        _SWAPPER_MODEL, providers=self._get_providers())
                except TypeError:
                    self.swapper = insightface.model_zoo.get_model(_SWAPPER_MODEL)
                print("[FaceSwapper] inswapper_128 loaded (PRIMARY — hard identity replacement).")
            except Exception as e:
                print(f"[FaceSwapper] inswapper_128 load failed: {type(e).__name__}: {e}")
                self.swapper = None
        else:
            print("[FaceSwapper] inswapper_128.onnx not found — trying SimSwap fallback.")

        # ── SimSwap (fallback — only loaded when inswapper unavailable) ──
        self.simswap   = None
        self.crossface = None
        if self.swapper is None and os.path.isfile(_SIMSWAP_MODEL) and os.path.isfile(_CROSSFACE_SIMSWAP):
            try:
                self.simswap   = _ort.InferenceSession(_SIMSWAP_MODEL,     providers=self._get_providers())
                self.crossface = _ort.InferenceSession(_CROSSFACE_SIMSWAP, providers=self._get_providers())
                _ep = self.simswap.get_providers()[0]
                print(f"[FaceSwapper] simswap_256 + crossface loaded on {_ep} (fallback).")
            except Exception as e:
                print(f"[FaceSwapper] SimSwap load failed: {type(e).__name__}: {e}")
                self.simswap = self.crossface = None

        # ── blendswap_256 (last resort — only when inswapper & SimSwap absent)
        self.blendswap = None
        if self.swapper is None and self.simswap is None and os.path.isfile(_BLENDSWAP_MODEL):
            _base = self._get_providers()
            _has_cuda = any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
                            for p in _base)
            _blend_providers = [(
                "CUDAExecutionProvider",
                {"device_id": 0,
                 "gpu_mem_limit": int(3.0 * 1024 * 1024 * 1024),
                 "arena_extend_strategy": "kSameAsRequested",
                 "cudnn_conv_algo_search": "DEFAULT"},
            )] if _has_cuda else ["CPUExecutionProvider"]
            try:
                self.blendswap = _ort.InferenceSession(_BLENDSWAP_MODEL, providers=_blend_providers)
                print(f"[FaceSwapper] blendswap_256 loaded on {self.blendswap.get_providers()[0]} (last resort).")
            except Exception as e:
                print(f"[FaceSwapper] blendswap_256 failed: {type(e).__name__}: {e}")

        if self.swapper is None and self.simswap is None and self.blendswap is None:
            print("[FaceSwapper] WARNING: no swap model available. Face swapping disabled.")

        # ── Face restorer: prefer CodeFormer, fall back to GFPGAN ──
        cf = CodeFormerRestorer()
        self.restorer = cf if cf.available else GFPGANRestorer()

    @staticmethod
    def _get_providers():
        """Returns ONNX Runtime execution providers (cached at first call)."""
        return resolve_providers()

    # ── blendswap_256 helpers ─────────────────────────────────────────

    @staticmethod
    def _warp_face(
        frame: np.ndarray,
        kps: np.ndarray,
        template: np.ndarray,
        size: tuple,
    ):
        """
        Estimate a 2-D similarity transform from the detected 5-point landmarks
        to a normalised template and warp the face region.

        Returns (crop, affine_2x3) or (None, None) on failure.
        """
        dst = template * np.array(size, dtype=np.float32)  # template → pixel coords
        affine, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float32),
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=100,
        )
        if affine is None:
            return None, None
        crop = cv2.warpAffine(
            frame, affine, size,
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_LINEAR,
        )
        return crop, affine

    def _blendswap_get(
        self,
        source_frame: np.ndarray,
        source_face,
        target_img: np.ndarray,
        target_face,
    ) -> np.ndarray:
        """
        Run blendswap_256 inference and paste the result back into source_frame.

        blendswap_256 terminology:
          'source' input (1,3,112,112) — identity donor  → target_face (template)
          'target' input (1,3,256,256) — canvas to modify → source_face (user)
        Output (1,3,256,256) RGB [0,1] — swapped 256×256 face.
        """
        h, w = source_frame.shape[:2]

        # --- Identity donor: template face at 112×112 (arcface_112_v2) ---
        id_crop, _ = self._warp_face(
            target_img, target_face.kps, _ARCFACE_112_V2, (112, 112)
        )
        if id_crop is None:
            print("[FaceSwapper] blendswap: identity warp failed — falling back.")
            return source_frame
        id_tensor = (
            id_crop[:, :, ::-1].astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[np.newaxis]  # (1,3,112,112) RGB [0,1]

        # --- Canvas: user face at 256×256 (ffhq_512) ---
        canvas_crop, affine = self._warp_face(
            source_frame, source_face.kps, _FFHQ_512, (256, 256)
        )
        if canvas_crop is None or affine is None:
            print("[FaceSwapper] blendswap: canvas warp failed — falling back.")
            return source_frame
        canvas_tensor = (
            canvas_crop[:, :, ::-1].astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[np.newaxis]  # (1,3,256,256) RGB [0,1]

        # --- ONNX inference ---
        try:
            output = self.blendswap.run(
                None, {"source": id_tensor, "target": canvas_tensor}
            )[0][0]  # (3,256,256) RGB [0,1]
        except Exception as e:
            print(f"[FaceSwapper] blendswap_256 inference failed: {e}")
            return source_frame

        # --- Convert output to BGR uint8 256×256 ---
        swapped_crop = np.clip(output, 0.0, 1.0).transpose(1, 2, 0)  # HWC RGB
        swapped_crop = (swapped_crop[:, :, ::-1] * 255.0).astype(np.uint8)  # BGR

        # --- Paste back into source_frame via inverse affine ---
        inv_affine = cv2.invertAffineTransform(affine)
        swapped_full = cv2.warpAffine(
            swapped_crop, inv_affine, (w, h), flags=cv2.INTER_LINEAR
        )

        # Build a smooth blend mask: erode to avoid border ringing, then Gaussian blur
        face_mask = cv2.warpAffine(
            np.ones((256, 256), dtype=np.float32), inv_affine, (w, h),
            flags=cv2.INTER_LINEAR,
        )
        k_erode = max(3, int(min(h, w) * 0.01)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
        face_mask = cv2.erode(face_mask, kernel, iterations=3)
        k_blur = max(21, int(min(h, w) * 0.04)) | 1
        face_mask = cv2.GaussianBlur(face_mask, (k_blur, k_blur), 0)
        face_mask = np.clip(face_mask, 0.0, 1.0)[:, :, np.newaxis]

        # Blend
        result = (
            swapped_full.astype(np.float32) * face_mask
            + source_frame.astype(np.float32) * (1.0 - face_mask)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    def _simswap_get(
        self,
        source_frame: np.ndarray,
        source_face,
        target_img: np.ndarray,
        target_face,
    ) -> np.ndarray:
        """
        Run SimSwap inference — hard identity replacement at 256×256.

        Pipeline:
          1. target_face (identity donor)  → arcface_112_v2 112×112
                                           → crossface_simswap → 512-d embedding (L2)
          2. source_face (canvas)          → arcface_112_v1 256×256
                                           → ImageNet normalise → NCHW float32
          3. simswap_256({'source': emb, 'target': crop}) → (1,3,256,256) [0,1]
          4. Paste back via inverse affine + soft mask.
        """
        h, w = source_frame.shape[:2]

        # ── identity embedding from target_face ─────────────────────────
        # SimSwap consumes a 512-d identity vector, but it was trained with a
        # different ArcFace than buffalo_l.  crossface (arcface_converter)
        # maps buffalo_l's 512-d normed_embedding into SimSwap's space.
        # NOTE: crossface expects a rank-2 (1,512) embedding, NOT an image.
        id_embedding = getattr(target_face, "normed_embedding", None)
        if id_embedding is None:
            print("[FaceSwapper] SimSwap: target face has no ArcFace embedding.")
            return source_frame
        id_embedding = id_embedding.reshape(1, -1).astype(np.float32)  # (1,512)

        try:
            cf_in = self.crossface.get_inputs()[0].name
            embedding = self.crossface.run(None, {cf_in: id_embedding})[0]  # (1,512)
        except Exception as e:
            print(f"[FaceSwapper] SimSwap crossface failed: {e}")
            return source_frame

        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = (embedding / (norm + 1e-8)).astype(np.float32)  # L2-normalised

        # ── canvas crop from source_face (user) ─────────────────────────
        canvas_crop, affine = self._warp_face(source_frame, source_face.kps,
                                              _ARCFACE_112_V1, (256, 256))
        if canvas_crop is None or affine is None:
            print("[FaceSwapper] SimSwap: canvas warp failed.")
            return source_frame

        canvas_rgb = canvas_crop[:, :, ::-1].astype(np.float32) / 255.0  # RGB [0,1]
        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        canvas_tensor = ((canvas_rgb - _mean) / _std).transpose(2, 0, 1)[np.newaxis]

        # ── SimSwap inference ────────────────────────────────────────────
        # Free PyTorch-reserved VRAM so ORT has room (prevents OOM on 4 GB).
        _free_torch_vram()
        try:
            output = self.simswap.run(
                None, {"source": embedding, "target": canvas_tensor}
            )[0][0]  # (3, 256, 256) float32 [0, 1]
        except Exception as e:
            print(f"[FaceSwapper] SimSwap inference failed: {e}")
            return source_frame

        # ── paste back ───────────────────────────────────────────────────
        swapped_crop = np.clip(output, 0.0, 1.0).transpose(1, 2, 0)      # HWC RGB
        swapped_crop = (swapped_crop[:, :, ::-1] * 255.0).astype(np.uint8)  # BGR

        inv_affine   = cv2.invertAffineTransform(affine)
        swapped_full = cv2.warpAffine(swapped_crop, inv_affine, (w, h),
                                      flags=cv2.INTER_LINEAR)

        face_mask = cv2.warpAffine(np.ones((256, 256), dtype=np.float32),
                                   inv_affine, (w, h), flags=cv2.INTER_LINEAR)
        k_erode = max(3, int(min(h, w) * 0.01)) | 1
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
        face_mask = cv2.erode(face_mask, kernel, iterations=3)
        k_blur  = max(21, int(min(h, w) * 0.04)) | 1
        face_mask = cv2.GaussianBlur(face_mask, (k_blur, k_blur), 0)
        face_mask = np.clip(face_mask, 0.0, 1.0)[:, :, np.newaxis]

        result = (swapped_full.astype(np.float32) * face_mask
                  + source_frame.astype(np.float32) * (1.0 - face_mask))
        return np.clip(result, 0, 255).astype(np.uint8)

    # ── Public API ────────────────────────────────────────────────────

    def swap(self, source_frame: np.ndarray, target_template_path: str, trace_faces: bool = False) -> np.ndarray:
        """
        Swaps the face in `source_frame` with the identity from a template file.
        This is the profile-based path (dropdown selection).

        Args:
            source_frame:  BGR image (numpy array) containing the user's face.
            target_template_path: Path to the target celebrity profile image.

        Returns:
            BGR image with the face swapped, or original frame on failure.
        """
        target_img = cv2.imread(target_template_path)
        if target_img is None:
            print(f"[FaceSwapper] Could not read template: {target_template_path}")
            return source_frame

        return self._do_swap(source_frame, target_img, trace_faces)

    def swap_with_target(self, source_frame: np.ndarray, target_img: np.ndarray,
                         trace_faces: bool = False, restore: bool = True,
                         verbose: bool = True) -> np.ndarray:
        """
        Swaps the face in `source_frame` with the identity from `target_img`.
        This is the custom-upload path (user-provided target face).

        Args:
            source_frame: BGR image containing the user's face.
            target_img:   BGR image containing the target face to apply.
            restore:      Run CodeFormer/GFPGAN HD restoration. Set False for
                          per-frame video processing (too slow / OOM-prone).
            verbose:      Print per-call diagnostic messages. Set False for
                          video frame loops to avoid terminal flooding.

        Returns:
            BGR image with the face swapped, or original frame on failure.
        """
        return self._do_swap(source_frame, target_img, trace_faces, restore, verbose)

    # ── Live webcam swap (fast path) ──────────────────────────────────

    def prepare_target_face(self, target_img: np.ndarray):
        """
        Detect and return the largest face in `target_img` as an InsightFace
        Face object (carries the identity embedding inswapper needs).

        Call this ONCE when a live session starts so the target identity is not
        re-detected on every webcam frame.  Returns None if no face is found.
        """
        if target_img is None:
            return None
        faces = self.app.get(target_img)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def swap_live(self, frame: np.ndarray, target_face) -> np.ndarray:
        """
        Low-latency swap for live webcam frames.

        Differences from the full pipeline (for speed / VRAM on a 4 GB GPU):
          • the target identity is pre-computed (prepare_target_face),
          • NO HD restoration / heavy post-processing,
          • every detected face in the frame is swapped to the target identity.

        Returns the swapped frame, or the original frame when no model / face.
        """
        if self.swapper is None or target_face is None or frame is None:
            return frame
        src_faces = self.app.get(frame)
        if not src_faces:
            return frame
        result = frame
        for sf in src_faces:
            result = self.swapper.get(result, sf, target_face, paste_back=True)
        return result

    # ── Internal ──────────────────────────────────────────────────────

    def _do_swap(self, source_frame: np.ndarray, target_img: np.ndarray,
                 trace_faces: bool = False, restore: bool = True, verbose: bool = True) -> np.ndarray:
        """
        Core swap logic with post-processing for clean output.
        Set verbose=False when calling per-frame (e.g. video) to suppress per-frame noise.
        """
        _p = print if verbose else (lambda *a, **k: None)
        _p("[FaceSwapper] Detecting faces...")
        source_faces = self.app.get(source_frame)
        target_faces = self.app.get(target_img)

        if not source_faces:
            _p("[FaceSwapper] No face detected in source image.")
            return source_frame
        if not target_faces:
            _p("[FaceSwapper] No face detected in target image.")
            return source_frame

        _p(f"[FaceSwapper] Found {len(source_faces)} source face(s), {len(target_faces)} target face(s).")

        source_face = max(source_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        target_face = max(target_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        if self.simswap is None and self.blendswap is None and self.swapper is None:
            _p("[FaceSwapper] No swap model loaded — using fallback.")
            return self._fallback_swap(source_frame, source_face, target_img, target_face)

        # ── Identity-similarity short-circuit ─────────────────────────
        # If the source and target are clearly the same person, skip the
        # swap entirely. inswapper_128 always reconstructs through a
        # 128x128 bottleneck, which subtly distorts features (e.g. nose
        # size) even when source==target. We compare the ArcFace identity
        # embeddings via cosine similarity. Threshold 0.55 = same person
        # with high confidence (typical false-match rate < 0.001%).
        identity_sim = self._cosine_similarity(
            getattr(source_face, "normed_embedding", None),
            getattr(target_face, "normed_embedding", None),
        )
        if identity_sim is not None:
            _p(f"[FaceSwapper] Identity similarity: {identity_sim:.3f}")
            if identity_sim >= 0.55:
                _p("[FaceSwapper] Source and target are the same identity — "
                   "skipping swap (only running HD restoration).")
                try:
                    return self._post_process(source_frame.copy(),
                                              source_frame, source_face)
                except Exception as e:
                    _p(f"[FaceSwapper] Restoration-only path failed: {e}")
                    return source_frame

        # ── Single-pass swap ──────────────────────────────────────────
        # Priority: inswapper_128 (strongest identity) → SimSwap → blendswap
        if self.swapper is not None:
            _p("[FaceSwapper] Running inswapper_128 (hard identity replacement)...")
            result = self.swapper.get(source_frame.copy(), source_face,
                                      target_face, paste_back=True)
        elif self.simswap is not None:
            _p("[FaceSwapper] Running SimSwap (256×256 fallback)...")
            result = self._simswap_get(source_frame, source_face, target_img, target_face)
        else:
            _p("[FaceSwapper] Running blendswap_256 (last resort)...")
            result = self._blendswap_get(source_frame, source_face, target_img, target_face)
        _p("[FaceSwapper] Swap complete.")

        # Lightweight post-processing (skipped per-frame for video speed)
        if restore:
            try:
                result = self._post_process(result, source_frame, source_face)
                _p("[FaceSwapper] Post-processing complete.")
            except Exception as e:
                _p(f"[FaceSwapper] Post-processing failed (returning raw swap): {e}")
                traceback.print_exc()

        if trace_faces:
            _p("[FaceSwapper] Applying face tracing visualization...")
            out_faces = self.app.get(result)
            if out_faces:
                for f in out_faces:
                    self._draw_face_tracing(result, f)

        return result

    def _draw_face_tracing(self, frame: np.ndarray, face) -> None:
        """Draws bounding box and landmarks on the frame. Tightly crops around face, ignoring hair."""
        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is not None:
            # Calculate a tight bounding box around the facial landmarks
            pts = landmarks.astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            
            # Draw Delaunay Triangulation Mesh
            img_rect = (0, 0, frame.shape[1], frame.shape[0])
            subdiv = cv2.Subdiv2D(img_rect)
            
            # Insert points
            for pt in pts:
                if 0 <= pt[0] < img_rect[2] and 0 <= pt[1] < img_rect[3]:
                    subdiv.insert((float(pt[0]), float(pt[1])))
            
            # Extract and draw triangles
            triangleList = subdiv.getTriangleList()
            pad = 20
            
            def in_face(p):
                return (x - pad <= p[0] <= x + w + pad) and (y - pad <= p[1] <= y + h + pad)
                
            for t in triangleList:
                pt1 = (int(t[0]), int(t[1]))
                pt2 = (int(t[2]), int(t[3]))
                pt3 = (int(t[4]), int(t[5]))
                
                # Filter out the bounding super-triangle vertices
                if in_face(pt1) and in_face(pt2) and in_face(pt3):
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(frame, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(frame, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Draw the landmark nodes
            for pt in pts:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            # Fallback if 106 landmarks aren't available
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            kps = getattr(face, 'kps', None)
            if kps is not None:
                for pt in kps:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    @staticmethod
    def _cosine_similarity(a, b):
        """Cosine similarity of two unit-norm vectors. Returns None on bad input."""
        if a is None or b is None:
            return None
        try:
            a = np.asarray(a, dtype=np.float32).ravel()
            b = np.asarray(b, dtype=np.float32).ravel()
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
            return float(np.dot(a, b) / denom)
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    #  POST-PROCESSING PIPELINE
    # ══════════════════════════════════════════════════════════════════

    def _post_process(self, swapped: np.ndarray, original: np.ndarray,
                      face) -> np.ndarray:
        """
        Quality-preserving post-processing:
          1. (Optional) GFPGAN HD face restoration on a padded face crop.
          2. Strict Anatomical Masking to ensure the swapped face DOES NOT
             bleed into the hair, ears, or background.
        """
        if getattr(self, "restorer", None) is not None and self.restorer.available:
            # Restore the swapped face to HD
            processed_face = self._restore_face_region(swapped, face)
        else:
            processed_face = swapped

        # Apply a strict anatomical mask to guarantee we only swap the inner face
        # and completely ignore the hair, ears, and background.
        strict_mask = self._build_strict_face_mask(original.shape[:2], face)
        mask_3c = strict_mask[:, :, np.newaxis]
        
        # Blend the processed face back into the purely original image
        final_blend = processed_face * mask_3c + original * (1.0 - mask_3c)
        return final_blend.astype(np.uint8)

    def _build_strict_face_mask(self, frame_shape: tuple, face) -> np.ndarray:
        """
        Builds a tight, highly constrained mask that strictly includes the inner
        facial features (eyes, nose, mouth, lower chin) but EXPLICITLY excludes 
        the forehead, jawline, ears, and hair.
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is not None and len(landmarks) == 106:
            # Expand the mask to capture the full natural shape of the source face
            # to prevent distortion or harsh seams on the cheeks.
            # 0-32: Full Jawline (captures the sides, cheeks, and chin perfectly)
            # 33-42: Eyebrows (caps the top, avoiding the forehead and bangs)
            selected_indices = list(range(0, 43))
            pts = landmarks[selected_indices].astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Dilate to ensure it smoothly reaches the very edge of the jawline
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Apply heavy blur for a seamless, soft blend into the original skin
            blur_size = max(31, int(min(h, w) * 0.08) | 1)
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        else:
            # Fallback if 106 landmarks aren't available
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            pad_x, pad_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
            cv2.rectangle(mask, (x1 + pad_x, y1 + pad_y),
                          (x2 - pad_x, y2 - pad_y), 255, -1)
            blur_size = max(31, int(min(h, w) * 0.08) | 1)
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask.astype(np.float32) / 255.0

    def _restore_face_region(self, frame: np.ndarray, face) -> np.ndarray:
        """
        Restore the swapped face WITHOUT distorting its geometry.

        CodeFormer / GFPGAN are trained on aligned 512×512 FFHQ crops. Feeding
        them a raw rectangular bbox crop (different aspect ratio) stretches the
        face when it is resized to a square — which is exactly what wrecks the
        eyes (skewed / hallucinated). The correct approach (roop / facefusion /
        Deep-Live-Cam) is:

          1. similarity-warp the face to the 512 FFHQ template using the 5-point
             keypoints, so it is perfectly aligned and square,
          2. run the restorer on that aligned crop,
          3. warp the restored crop back with the INVERSE affine,
          4. blend through a soft, slightly inset face mask.
        """
        h, w = frame.shape[:2]
        kps = getattr(face, "kps", None)
        if kps is None:
            return frame

        # 1. Align the face to a 512×512 FFHQ canvas (no stretching).
        aligned, affine = self._warp_face(frame, kps, _FFHQ_512, (512, 512))
        if aligned is None or affine is None:
            return frame

        # 2. Restore the aligned, geometry-correct crop.
        try:
            restored = self.restorer.restore(aligned)
        except Exception as e:
            print(f"[FaceSwapper] Restoration failed: {e}")
            return frame
        if restored is None or restored.shape[:2] != (512, 512):
            return frame

        # 3. Soft elliptical mask in aligned space (inset so the crop border
        #    never shows; feathered so the paste-back is seamless).
        mask = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(mask, (256, 262), (208, 244), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=18, sigmaY=18)

        # 4. Warp restored crop + mask back into the full frame via inverse affine.
        restored_back = cv2.warpAffine(
            restored, affine, (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask_back = cv2.warpAffine(
            mask, affine, (w, h),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        mask_back = np.clip(mask_back, 0.0, 1.0)[:, :, np.newaxis]

        out = (restored_back.astype(np.float32) * mask_back +
               frame.astype(np.float32) * (1.0 - mask_back))
        return out.astype(np.uint8)

    def _correct_luminance_to_border(self, swapped: np.ndarray,
                                      original: np.ndarray,
                                      face_mask: np.ndarray) -> np.ndarray:
        """
        Matches the swapped face's luminance (L channel in LAB) to the
        brightness of the SURROUNDING skin border, not the original face.

        This fixes the pale-face-on-dark-body problem without reverting
        the identity transfer.

        Border ring = dilated_mask − face_mask (a ring of pixels around the face).
        """
        if swapped.shape != original.shape:
            return swapped

        # Build the border ring mask
        hard_mask = (face_mask > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate(hard_mask, kernel, iterations=2)
        border_ring = dilated - hard_mask  # ring of surrounding skin

        if border_ring.sum() < 50 or hard_mask.sum() < 50:
            return swapped

        # Convert to LAB
        swapped_lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Only correct the L (luminance) channel — preserve A and B (colour)
        # so the target identity's skin colour is kept
        L_swap = swapped_lab[:, :, 0]
        L_border = original_lab[:, :, 0]

        # Stats from the face region and the border ring
        face_L_mean = L_swap[hard_mask == 1].mean()
        face_L_std = L_swap[hard_mask == 1].std() + 1e-6
        border_L_mean = L_border[border_ring == 1].mean()
        border_L_std = L_border[border_ring == 1].std() + 1e-6

        # Shift luminance: face brightness → border brightness
        corrected_L = (L_swap - face_L_mean) * (border_L_std / face_L_std) + border_L_mean
        corrected_L = np.clip(corrected_L, 0, 255)

        # Apply only within the face mask with soft blending
        swapped_lab[:, :, 0] = corrected_L * face_mask + L_swap * (1.0 - face_mask)

        return cv2.cvtColor(swapped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _seamless_clone_eroded(self, swapped: np.ndarray,
                                original: np.ndarray,
                                face_mask: np.ndarray) -> np.ndarray:
        """
        Poisson seamless clone with an eroded mask.  The erosion ensures
        only the BOUNDARY is affected by seamless blending — the interior
        keeps the full swapped identity.
        """
        try:
            # Create a hard mask and erode it so seamlessClone only
            # fixes the edges, not the whole face interior
            hard_mask = ((face_mask > 0.3) * 255).astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            eroded_mask = cv2.erode(hard_mask, kernel, iterations=3)

            if eroded_mask.sum() < 100:
                return swapped

            # Centroid for seamlessClone
            moments = cv2.moments(eroded_mask)
            if moments["m00"] < 1:
                return swapped
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            mask_3c = cv2.merge([eroded_mask, eroded_mask, eroded_mask])
            result = cv2.seamlessClone(swapped, original, mask_3c,
                                       (cx, cy), cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            print(f"[FaceSwapper] seamlessClone skipped: {e}")
            return swapped

    # ── Mask construction ────────────────────────────────────────────

    def _build_face_mask(self, frame_shape: tuple, face) -> np.ndarray:
        """
        Creates a soft (feathered) mask from a convex hull of all available
        landmarks.  Returns float32 in [0, 1].
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is None:
            landmarks = getattr(face, 'kps', None)

        if landmarks is not None and len(landmarks) > 2:
            pts = landmarks.astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            pad_x, pad_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
            cv2.rectangle(mask, (x1 + pad_x, y1 + pad_y),
                          (x2 - pad_x, y2 - pad_y), 255, -1)

        # Dilate to ensure we cover the full boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Heavy blur for soft falloff
        blur_size = max(51, int(min(h, w) * 0.08) | 1)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return mask.astype(np.float32) / 255.0

    def _build_zone_masks(self, frame_shape: tuple, face) -> dict:
        """
        Builds individual soft masks for each facial zone (forehead, cheeks,
        nose, chin, mouth) using the 106-landmark topology.
        Falls back to an empty dict if dense landmarks are unavailable.
        """
        landmarks = getattr(face, 'landmark_2d_106', None)
        if landmarks is None or len(landmarks) < 106:
            return {}

        h, w = frame_shape
        pts_all = landmarks.astype(np.int32)
        zones = {}

        # ── Forehead (synthesised) ────────────────────────────────────
        # The 106-landmark set doesn't explicitly include the forehead.
        # We construct it by taking the eyebrow points and shifting them
        # upward by 60% of the brow-to-chin distance.
        try:
            brow_pts = pts_all[list(range(33, 43))]
            chin_y = pts_all[16][1]  # bottom of chin (landmark 16)
            brow_y_avg = brow_pts[:, 1].mean()
            forehead_shift = int((chin_y - brow_y_avg) * 0.45)

            forehead_pts = brow_pts.copy()
            forehead_pts[:, 1] -= forehead_shift
            # Combine with original brow points to form a closed region
            combined = np.vstack([brow_pts, forehead_pts[::-1]])
            hull = cv2.convexHull(combined)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            zones["forehead"] = mask.astype(np.float32) / 255.0
        except Exception:
            pass

        # ── Standard zones ────────────────────────────────────────────
        for zone_name, zone_def in _FACE_ZONES_106.items():
            if zone_name == "forehead":
                continue
            try:
                idx_list = zone_def.get("pts", [])
                if not idx_list:
                    continue
                zone_pts = pts_all[idx_list]
                if len(zone_pts) < 3:
                    continue

                hull = cv2.convexHull(zone_pts)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                zones[zone_name] = mask.astype(np.float32) / 255.0
            except Exception:
                continue

        return zones

    # ── Denoising ────────────────────────────────────────────────────

    def _denoise_face(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Bilateral filter constrained to the face.  Removes high-frequency
        swap artefacts while preserving edges and skin pores.

        Math:  BF(I)(x) = (1/W) Σ G_σs(‖x−y‖) · G_σr(|I(x)−I(y)|) · I(y)
        """
        filtered = cv2.bilateralFilter(frame, d=9, sigmaColor=60, sigmaSpace=60)
        mask_3c = mask[:, :, np.newaxis]
        return (filtered * mask_3c + frame * (1.0 - mask_3c)).astype(np.uint8)

    # ── Colour correction ────────────────────────────────────────────

    def _correct_colour_per_zone(self, swapped: np.ndarray, original: np.ndarray,
                                 full_mask: np.ndarray,
                                 zone_masks: dict) -> np.ndarray:
        """
        Per-region colour correction in LAB space.  Each facial zone
        (forehead, left cheek, right cheek, nose, chin) is corrected
        independently so that lighting gradients are preserved rather
        than averaged out.

        For each zone and each LAB channel:
            corrected = (pixel − μ_swap) × (σ_orig / σ_swap) + μ_orig
        """
        if swapped.shape != original.shape:
            return swapped

        swapped_lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
        original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

        if zone_masks:
            # Per-zone correction
            for zone_name, z_mask in zone_masks.items():
                bin_z = (z_mask > 0.5).astype(np.uint8)
                if bin_z.sum() < 50:
                    continue
                for c in range(3):
                    self._transfer_channel(swapped_lab[:, :, c],
                                           original_lab[:, :, c],
                                           bin_z, z_mask)
        else:
            # Fallback: global correction using full mask
            bin_mask = (full_mask > 0.5).astype(np.uint8)
            if bin_mask.sum() >= 100:
                for c in range(3):
                    self._transfer_channel(swapped_lab[:, :, c],
                                           original_lab[:, :, c],
                                           bin_mask, full_mask)

        return cv2.cvtColor(swapped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _transfer_channel(src_ch: np.ndarray, ref_ch: np.ndarray,
                          bin_mask: np.ndarray, soft_mask: np.ndarray):
        """
        In-place colour transfer for a single LAB channel within a masked zone.
        """
        src_mean = src_ch[bin_mask == 1].mean()
        src_std = src_ch[bin_mask == 1].std() + 1e-6
        ref_mean = ref_ch[bin_mask == 1].mean()
        ref_std = ref_ch[bin_mask == 1].std() + 1e-6

        corrected = (src_ch - src_mean) * (ref_std / src_std) + ref_mean
        corrected = np.clip(corrected, 0, 255)
        src_ch[:] = corrected * soft_mask + src_ch * (1.0 - soft_mask)

    # ── Laplacian pyramid blending ───────────────────────────────────

    def _laplacian_blend(self, swapped: np.ndarray, original: np.ndarray,
                         mask: np.ndarray, levels: int = 5) -> np.ndarray:
        """
        Multi-band blending using Laplacian pyramids.  Each frequency band
        of the swapped and original images is blended independently using the
        Gaussian pyramid of the mask.  This produces far smoother transitions
        than a single alpha blend because low-frequency colour transitions are
        handled separately from high-frequency texture.

        Math per level l:
            L_blend(l) = G_mask(l) · L_swap(l) + (1 − G_mask(l)) · L_orig(l)
        """
        # Ensure dimensions are suitable for the pyramid (divisible by 2^levels)
        h, w = swapped.shape[:2]
        factor = 2 ** levels
        new_h = (h // factor) * factor
        new_w = (w // factor) * factor

        if new_h != h or new_w != w:
            s_img = swapped[:new_h, :new_w]
            o_img = original[:new_h, :new_w]
            m_img = mask[:new_h, :new_w]
        else:
            s_img = swapped
            o_img = original
            m_img = mask

        s_f = s_img.astype(np.float32)
        o_f = o_img.astype(np.float32)

        # Build Gaussian pyramids
        gp_s = [s_f]
        gp_o = [o_f]
        gp_m = [m_img]

        for _ in range(levels):
            gp_s.append(cv2.pyrDown(gp_s[-1]))
            gp_o.append(cv2.pyrDown(gp_o[-1]))
            gp_m.append(cv2.pyrDown(gp_m[-1]))

        # Build Laplacian pyramids
        lp_s = []
        lp_o = []
        for i in range(levels):
            up_s = cv2.pyrUp(gp_s[i + 1], dstsize=(gp_s[i].shape[1], gp_s[i].shape[0]))
            up_o = cv2.pyrUp(gp_o[i + 1], dstsize=(gp_o[i].shape[1], gp_o[i].shape[0]))
            lp_s.append(gp_s[i] - up_s)
            lp_o.append(gp_o[i] - up_o)
        lp_s.append(gp_s[levels])
        lp_o.append(gp_o[levels])

        # Blend each level using the mask pyramid
        lp_blend = []
        for i in range(levels + 1):
            m = gp_m[i][:, :, np.newaxis] if m_img.ndim == 2 or gp_m[i].ndim == 2 else gp_m[i]
            if m.ndim == 2:
                m = m[:, :, np.newaxis]
            blended_level = lp_s[i] * m + lp_o[i] * (1.0 - m)
            lp_blend.append(blended_level)

        # Reconstruct from the blended Laplacian pyramid
        reconstructed = lp_blend[levels]
        for i in range(levels - 1, -1, -1):
            reconstructed = cv2.pyrUp(reconstructed,
                                      dstsize=(lp_blend[i].shape[1], lp_blend[i].shape[0]))
            reconstructed += lp_blend[i]

        # Place back into full-size frame if we cropped
        result = np.clip(reconstructed, 0, 255).astype(np.uint8)
        if new_h != h or new_w != w:
            full = original.copy()
            full[:new_h, :new_w] = result
            return full

        return result

    # ── Seamless clone pass ──────────────────────────────────────────

    def _seamless_clone_pass(self, blended: np.ndarray, original: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
        """
        Final Poisson-equation boundary pass using cv2.seamlessClone.
        Solves for pixel values that satisfy Laplace's equation at the
        boundary, producing a mathematically perfect colour transition.
        """
        try:
            # Build a hard mask for seamlessClone (it needs uint8 0/255)
            hard_mask = (mask > 0.3).astype(np.uint8) * 255

            # Erode slightly so seamlessClone doesn't pull in background pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            hard_mask = cv2.erode(hard_mask, kernel, iterations=2)

            # Compute the centroid of the mask as the clone centre
            moments = cv2.moments(hard_mask)
            if moments["m00"] < 1:
                return blended
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Ensure the mask is 3-channel for seamlessClone
            if hard_mask.ndim == 2:
                hard_mask_3c = cv2.merge([hard_mask, hard_mask, hard_mask])
            else:
                hard_mask_3c = hard_mask

            result = cv2.seamlessClone(blended, original, hard_mask_3c,
                                       (cx, cy), cv2.MIXED_CLONE)
            return result
        except Exception as e:
            print(f"[FaceSwapper] seamlessClone pass skipped: {e}")
            return blended

    # ── Sharpening ───────────────────────────────────────────────────

    def _sharpen(self, frame: np.ndarray, mask: np.ndarray,
                 amount: float = 0.25) -> np.ndarray:
        """
        Unsharp mask confined to the face region.
            sharpened = original + amount × (original − blurred)
        """
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

        mask_3c = mask[:, :, np.newaxis]
        result = (sharpened.astype(np.float32) * mask_3c +
                  frame.astype(np.float32) * (1.0 - mask_3c))
        return np.clip(result, 0, 255).astype(np.uint8)

    # ── Fallback ─────────────────────────────────────────────────────

    def _fallback_swap(self, source_frame, source_face, target_img, target_face):
        """
        Basic fallback if inswapper_128.onnx is not available.
        Uses simple crop + resize + Poisson blending (similar to original code).
        """
        try:
            # Extract bounding boxes
            sx1, sy1, sx2, sy2 = [int(v) for v in source_face.bbox]
            tx1, ty1, tx2, ty2 = [int(v) for v in target_face.bbox]

            # Crop target face and resize to source face dimensions
            target_crop = target_img[ty1:ty2, tx1:tx2]
            sw, sh = sx2 - sx1, sy2 - sy1
            target_resized = cv2.resize(target_crop, (sw, sh))

            # Create mask for Poisson blending
            mask = 255 * np.ones(target_resized.shape, dtype=target_resized.dtype)
            center = (sx1 + sw // 2, sy1 + sh // 2)

            output = cv2.seamlessClone(target_resized, source_frame, mask, center, cv2.NORMAL_CLONE)
            return output
        except Exception as e:
            print(f"[FaceSwapper] Fallback blending failed: {e}")
            return source_frame