"""
backend/__init__.py
Package marker for the backend module.
"""

# ── Compatibility patch: coqui-tts + transformers 5.x ────────────────
# Must be applied before any TTS import. The function was removed in
# transformers 5.x but coqui-tts still references it.
try:
    import torch
    import transformers.pytorch_utils as _pt_utils
    if not hasattr(_pt_utils, 'isin_mps_friendly'):
        _pt_utils.isin_mps_friendly = torch.isin
except Exception:
    pass
