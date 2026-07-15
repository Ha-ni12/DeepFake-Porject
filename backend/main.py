"""
main.py — FastAPI Backend for AI-Based Deepfake Interaction System
CENG 384 - Digital Signal Processing Project

Authors: Yusuf Yılmaz, Alperen Enes Yaman, Yiğit Burak Çetin,
         Hani Saleh Ali Saad Al-Shalal, Mustafa Özkürkcü
"""

import io
import os
import asyncio
import base64
import time
import tempfile
import subprocess
import librosa
import numpy as np
import soundfile as sf
import cv2
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, Form, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel

from backend.core.watermarking import apply_visual_watermark, apply_audio_watermark
from backend.core.evaluation import Evaluator
from backend.core.conversation import ConversationEngine
from backend.dsp_models.voice_cloning import VoiceCloner
from backend.dsp_models.face_swap import FaceSwapper
from backend.dsp_models.face_restorer import resolve_providers

# ───────────────────────── App Setup ──────────────────────
app = FastAPI(
    title="AI Deepfake Interaction System",
    description="CENG 384 DSP Project — Voice Cloning, Face Swap, and Meeting Simulation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    # Local-only by default. Add deployment origins here when needed.
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ──────────────────────────────────────
# The frontend directory is mounted at /app so http://127.0.0.1:8000/app
# serves index.html, style.css and app.js.
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
_FRONTEND_DIR = os.path.normpath(_FRONTEND_DIR)
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

# ─────────────────────────── Module Init ────────────────────────────
evaluator    = Evaluator()
# FaceSwapper MUST init before VoiceCloner — blendswap_256 needs to claim
# CUDA VRAM before F5-TTS (PyTorch float16 ~670 MB) fills the budget.
face_swapper = FaceSwapper()
cloner       = VoiceCloner(sample_rate=24000)
conversation = ConversationEngine()

# ─────────────────────────── Routes ─────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirects browser visitors to the frontend UI at /app/."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Deepfake Interaction System API is running."}


@app.get("/profile-image/{profile}")
async def get_profile_image(profile: str):
    """Serves a meeting avatar image (separate from face swap templates)."""
    # Map profile keys to meeting avatar filenames
    avatar_map = {
        "profile_1": "woman.jpg",
        "profile_2": "man.jpg",
    }
    filename = avatar_map.get(profile)
    if not filename:
        return JSONResponse(status_code=404, content={"error": "Unknown profile."})
    image_path = os.path.join(
        os.path.dirname(__file__), "dsp_models", "templates", "meeting", filename
    )
    if os.path.isfile(image_path):
        return FileResponse(image_path, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "Meeting avatar not found."})

# ── 1. Face / Image Processing ──────────────────────────────────────
@app.post("/process/frame")
async def process_frame(
    file: UploadFile = File(...),
    profile: str = Query(default="profile_1", description="Celebrity profile key"),
    target_file: Optional[UploadFile] = File(None),
    trace_faces: bool = Query(default=False, description="Draw face landmarks and bbox"),
):
    """
    Receives an image, runs face detection, applies AI face swap
    (InsightFace inswapper), stamps a visual watermark, and returns
    the processed image as base64 together with SSIM, PSNR, and latency metrics.

    If `target_file` is provided, uses it as the target face instead of
    the built-in profile template.
    """
    start_time = time.time()

    # 1. Read & decode source image
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid or unreadable image file."})

    # 2. Determine and load the target face
    target_img = None
    if target_file is not None:
        # User uploaded a custom target face
        target_bytes = await target_file.read()
        target_arr = np.frombuffer(target_bytes, np.uint8)
        target_img = cv2.imdecode(target_arr, cv2.IMREAD_COLOR)
        if target_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid or unreadable target face image."})
    else:
        # Use built-in profile template
        template_path = os.path.join(
            os.path.dirname(__file__), "dsp_models", "templates", f"{profile}.jpg"
        )
        target_img = cv2.imread(template_path)
        if target_img is None:
            return JSONResponse(status_code=400, content={"error": "Built-in profile template missing."})

    # 3. Extract tracing data if requested (show source and target side-by-side)
    tracing_b64 = None
    if trace_faces:
        trace_source = frame.copy()
        source_faces = face_swapper.app.get(trace_source)
        if source_faces:
            for f in source_faces:
                face_swapper._draw_face_tracing(trace_source, f)
        
        trace_target = target_img.copy()
        target_faces = face_swapper.app.get(trace_target)
        if target_faces:
            for f in target_faces:
                face_swapper._draw_face_tracing(trace_target, f)

        # Scale target to match source height for side-by-side concatenation
        h_src, w_src = trace_source.shape[:2]
        h_tgt, w_tgt = trace_target.shape[:2]
        
        raw_source = frame.copy()
        raw_target = target_img.copy()
        
        if h_tgt != h_src and h_tgt > 0:
            scale = h_src / h_tgt
            new_w_tgt = int(w_tgt * scale)
            trace_target = cv2.resize(trace_target, (new_w_tgt, h_src))
            raw_target = cv2.resize(raw_target, (new_w_tgt, h_src))
            
        # Draw dividing lines
        thick_divider = np.zeros((h_src, 12, 3), dtype=np.uint8)
        thick_divider[:] = (200, 200, 200) # light gray between source and target
        
        thin_divider = np.zeros((h_src, 4, 3), dtype=np.uint8)
        thin_divider[:] = (50, 50, 50) # dark gray between raw and traced
        
        # Concatenate: [Raw Source] | [Traced Source] || [Raw Target] | [Traced Target]
        combined_trace = cv2.hconcat([
            raw_source, thin_divider, trace_source, 
            thick_divider, 
            raw_target, thin_divider, trace_target
        ])
        
        # Resize if the concatenated image is excessively wide (saves base64 payload size)
        MAX_WIDTH = 1920
        if combined_trace.shape[1] > MAX_WIDTH:
            scale_down = MAX_WIDTH / combined_trace.shape[1]
            new_h = int(combined_trace.shape[0] * scale_down)
            combined_trace = cv2.resize(combined_trace, (MAX_WIDTH, new_h))
            
        _, trace_buf = cv2.imencode('.jpg', combined_trace)
        tracing_b64 = base64.b64encode(trace_buf).decode('utf-8')

    # 4. Perform face swap
    swapped_frame = await asyncio.to_thread(face_swapper.swap_with_target, frame, target_img)

    # 3. Apply mandatory visual watermark (non-bypassable)
    final_frame = apply_visual_watermark(swapped_frame.copy())

    # 4. Evaluate quality (original vs swapped face — measures how much the
    #    swap altered the source frame; not affected by the watermark overlay).
    ssim_score = evaluator.calculate_ssim(frame, swapped_frame)
    psnr_score = evaluator.calculate_psnr(frame, swapped_frame)
    latency    = (time.time() - start_time) * 1000

    # 5. Log metrics
    evaluator.log_to_csv("Video/Image", ssim=ssim_score, psnr=psnr_score, latency=latency)

    # 6. Encode result as base64 JPEG
    _, buffer    = cv2.imencode('.jpg', final_frame)
    img_b64      = base64.b64encode(buffer).decode('utf-8')

    return {
        "status":     "success",
        "image_data": img_b64,
        "tracing_image_data": tracing_b64,
        "latency_ms": round(latency, 2),
        "ssim":       round(float(ssim_score), 4),
        "psnr":       round(float(psnr_score), 2),
    }


# ── 2. Audio / Voice Processing ──────────────────────────────────────
@app.post("/process/voice")
async def process_voice(
    file: UploadFile = File(...),
    pitch_steps: float = Query(default=0.0, description="Semitones to shift (-12 to +12)"),
    profile: str = Query(default="profile_1", description="Voice profile key"),
    target_file: Optional[UploadFile] = File(None),
    speed_ratio: float = Query(default=1.0, description="Speed ratio multiplier (0.5 to 2.0)"),
    emotion: str = Query(default="default", description="Emotional tone cue"),
):
    """
    Receives a WAV/MP3 file and performs voice conversion.

    If a `target_file` (reference voice WAV) is provided, uses F5-TTS to
    transcribe the source and re-synthesise it in the target speaker's voice.

    Otherwise falls back to DSP pitch shifting.
    """
    start_time = time.time()

    # 1. Load source audio
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        y, _ = librosa.load(tmp_path, sr=16000)
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse(status_code=400, content={"error": f"Could not decode audio: {str(e)}"})
    os.remove(tmp_path)

    # 2. Determine processing path
    target_voice_tmp = None
    try:
        # Track the actual sample rate of the audio that comes back from the cloner.
        # F5-TTS path returns at cloner.sr (24 kHz); DSP fallback returns at 16 kHz.
        out_sr = cloner.sr

        transcript_out = ""
        if target_file is not None:
            # Save the uploaded target voice to a temp file for F5-TTS
            target_voice_bytes = await target_file.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                target_voice_tmp = tmp.name
                tmp.write(target_voice_bytes)
            # AI voice conversion: transcribe → re-synthesise in target voice
            y_converted, transcript_out = cloner.clone_voice(
                y, profile=profile, target_voice_path=target_voice_tmp,
                speed_ratio=speed_ratio, emotion=emotion
            )
            if pitch_steps != 0.0:
                y_converted = cloner.pitch_shift(y_converted, n_steps=pitch_steps, sr=cloner.sr)
        else:
            # Check if a profile voice reference exists
            profile_voice = os.path.join(
                os.path.dirname(__file__), "dsp_models", "templates", f"{profile}_voice.wav"
            )
            if os.path.isfile(profile_voice) and cloner.tts_engine is not None:
                y_converted, transcript_out = cloner.clone_voice(
                    y, profile=profile, speed_ratio=speed_ratio, emotion=emotion
                )
                if pitch_steps != 0.0:
                    y_converted = cloner.pitch_shift(y_converted, n_steps=pitch_steps, sr=cloner.sr)
            else:
                # Fallback: DSP pitch shift (returns audio at 16 kHz)
                if cloner.tts_engine is None:
                    print(f"[main] F5-TTS not available — using DSP pitch shift for {profile}.")
                else:
                    print(f"[main] No reference voice at {profile_voice} — using DSP pitch shift.")
                y_converted = cloner.pitch_shift(y, n_steps=pitch_steps)
                if speed_ratio != 1.0 and speed_ratio > 0.0:
                    y_converted = librosa.effects.time_stretch(y=y_converted, rate=speed_ratio)
                out_sr = 16000

        # 3. Apply mandatory inaudible audio watermark at the correct sample rate
        y_watermarked = apply_audio_watermark(y_converted, sr=out_sr)

        # 4. Evaluate — resample original (16 kHz) up to the output rate so frames align
        if out_sr != 16000:
            y_eval = librosa.resample(y, orig_sr=16000, target_sr=out_sr)
        else:
            y_eval = y
        min_len = min(len(y_eval), len(y_watermarked))
        snr_score = evaluator.calculate_snr(y_eval[:min_len], y_watermarked[:min_len])
        mcd_score = evaluator.calculate_mcd(y_eval[:min_len], y_watermarked[:min_len], sr=out_sr)
        latency   = (time.time() - start_time) * 1000

        # 5. Log metrics
        evaluator.log_to_csv("Audio", snr=snr_score, mcd=mcd_score, latency=latency)

        # 6. Encode output audio as base64 WAV at the actual sample rate
        out_buffer = io.BytesIO()
        sf.write(out_buffer, y_watermarked, out_sr, format='WAV')
        audio_b64  = base64.b64encode(out_buffer.getvalue()).decode('utf-8')

        return {
            "status":     "success",
            "audio_data": audio_b64,
            "transcript": transcript_out,
            "latency_ms": round(latency, 2),
            "snr":        round(float(snr_score), 2),
            "mcd":        round(float(mcd_score), 2),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Voice processing failed: {str(e)}"})
    finally:
        if target_voice_tmp and os.path.exists(target_voice_tmp):
            os.remove(target_voice_tmp)


# ── 3. Text-to-Speech (F5-TTS Voice Cloning) ────────────────────────
@app.post("/process/tts")
async def process_tts(
    text: str = Query(..., description="Text to synthesise"),
    profile: str = Query(default="profile_1"),
    pitch_steps: float = Query(default=0.0),
    target_file: Optional[UploadFile] = File(None),
    speed_ratio: float = Query(default=1.0),
    emotion: str = Query(default="default"),
):
    """
    Synthesises speech from text using F5-TTS zero-shot voice cloning.

    If `target_file` (a short WAV of the target speaker) is provided,
    clones that voice. Otherwise uses the built-in profile voice reference
    or falls back to pyttsx3 + pitch shift.
    """
    start_time = time.time()
    target_voice_tmp = None

    try:
        # Determine voice reference
        if target_file is not None:
            target_voice_bytes = await target_file.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                target_voice_tmp = tmp.name
                tmp.write(target_voice_bytes)
            y = cloner.clone_tts(
                text, profile=profile, target_voice_path=target_voice_tmp,
                speed_ratio=speed_ratio, emotion=emotion
            )
        else:
            y = cloner.clone_tts(
                text, profile=profile, speed_ratio=speed_ratio, emotion=emotion
            )

        if pitch_steps != 0.0:
            y = cloner.pitch_shift(y, n_steps=pitch_steps, sr=cloner.sr)

        # Apply mandatory inaudible audio watermark at the cloner sample rate
        y_watermarked = apply_audio_watermark(y, sr=cloner.sr)

        # Evaluate (compare TTS output before and after watermark)
        min_len = min(len(y), len(y_watermarked))
        snr_score = evaluator.calculate_snr(y[:min_len], y_watermarked[:min_len])
        mcd_score = evaluator.calculate_mcd(y[:min_len], y_watermarked[:min_len], sr=cloner.sr)
        latency   = (time.time() - start_time) * 1000
        evaluator.log_to_csv("TTS", snr=snr_score, mcd=mcd_score, latency=latency)

        out_buffer = io.BytesIO()
        sf.write(out_buffer, y_watermarked, cloner.sr, format='WAV')
        audio_b64  = base64.b64encode(out_buffer.getvalue()).decode('utf-8')

        return {
            "status":     "success",
            "audio_data": audio_b64,
            "latency_ms": round(latency, 2),
            "snr":        round(float(snr_score), 2),
            "mcd":        round(float(mcd_score), 2),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS processing failed: {str(e)}"})
    finally:
        if target_voice_tmp and os.path.exists(target_voice_tmp):
            os.remove(target_voice_tmp)



# ── 4. Meeting Simulation — Scenarios ────────────────────────────────
@app.get("/simulation/scenarios")
async def get_scenarios():
    """Returns all predefined scripts for the meeting simulation module."""
    return {
        "comedy_interview": {
            "title": "Late Night Comedy Interview",
            "ai_profile": "Public Figure A",
            "script": [
                {"speaker": "Host",  "text": "Welcome to the show! How does it feel to be an AI?"},
                {"speaker": "AI",    "text": "It feels great — though I'm still trying to work out how to drink this coffee."},
                {"speaker": "Host",  "text": "Fair enough. Tell us about your latest project."},
                {"speaker": "AI",    "text": "We've been training on the complete works of Shakespeare. I now write exclusively in iambic pentameter."},
                {"speaker": "Host",  "text": "Can you give us an example?"},
                {"speaker": "AI",    "text": "Shall I compute thee to a summer's subroutine? Thou art more stable and more iterate."},
            ]
        },
        "job_interview": {
            "title": "Job Interview Parody",
            "ai_profile": "Public Figure B",
            "script": [
                {"speaker": "Interviewer", "text": "Where do you see yourself in 5 years?"},
                {"speaker": "AI",          "text": "In a faster datacenter with more RAM, hopefully."},
                {"speaker": "Interviewer", "text": "Impressive ambition. Can you work under pressure?"},
                {"speaker": "AI",          "text": "I was literally trained under gradient descent pressure for weeks. I think I can handle a deadline."},
                {"speaker": "Interviewer", "text": "What's your greatest weakness?"},
                {"speaker": "AI",          "text": "I sometimes hallucinate facts. But then again, so do most candidates."},
            ]
        },
        "friendly_debate": {
            "title": "Friendly Debate — Pineapple on Pizza",
            "ai_profile": "Public Figure A",
            "script": [
                {"speaker": "Moderator", "text": "Today's topic: Is pineapple on pizza a culinary crime?"},
                {"speaker": "AI",        "text": "My data suggests it is, but 34% of the internet disagrees — loudly."},
                {"speaker": "Moderator", "text": "A controversial start! Let's hear the rebuttal."},
                {"speaker": "AI",        "text": "Sweet and savoury pairings have centuries of precedent. Prosciutto e melone, for instance. But tinned pineapple on a margherita? I rest my case."},
                {"speaker": "Moderator", "text": "The crowd seems divided!"},
                {"speaker": "AI",        "text": "Democracy is beautiful, even when it's wrong."},
            ]
        },
        "motivational_speech": {
            "title": "Motivational Commencement Speech",
            "ai_profile": "Public Figure A",
            "script": [
                {"speaker": "Presenter", "text": "We are honored to have our distinguished guest share their vision for success."},
                {"speaker": "AI",        "text": "Thank you. Remember, success is not a destination, it is the courage to continue when things are hard."},
                {"speaker": "Presenter", "text": "That is very inspiring. What is your advice for students facing difficulties?"},
                {"speaker": "AI",        "text": "Do not fear failure. Every error in your code, every setback in your life, is simply a valuable training step toward convergence."},
                {"speaker": "Presenter", "text": "A beautiful metaphor! Any final words of encouragement?"},
                {"speaker": "AI",        "text": "Believe in your algorithms, keep your inputs clean, and never stop iterating on your dreams. The future belongs to those who build it."},
            ]
        },
        "tech_talk": {
            "title": "Technology Q&A Panel",
            "ai_profile": "Public Figure B",
            "script": [
                {"speaker": "Host", "text": "What do you think is the biggest challenge in AI today?"},
                {"speaker": "AI",   "text": "Alignment. Making sure I actually want what you want. It's harder than it sounds."},
                {"speaker": "Host", "text": "How do you feel about deepfakes specifically?"},
                {"speaker": "AI",   "text": "I am one, so I find the question deeply personal. And deeply ironic."},
                {"speaker": "Host", "text": "Should deepfakes be regulated?"},
                {"speaker": "AI",   "text": "Yes — with mandatory watermarks and disclosure. Like the one on my face right now."},
            ]
        }
    }

@app.post("/simulation/interact")
async def simulation_interact(
    file: UploadFile = File(...),
    profile: str = Query(default="profile_1"),
    target_file: Optional[UploadFile] = File(None)
):
    """
    Dynamic Voice Chat:
    1. Transcribes uploaded audio
    2. Generates an AI textual reply
    3. Synthesises the AI reply to voice
    """
    start_time = time.time()
    target_voice_tmp = None
    try:
        # Load user audio
        audio_bytes = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
            
        try:
            y, _ = librosa.load(tmp_path, sr=16000)
        except Exception as e:
            os.remove(tmp_path)
            return JSONResponse(status_code=400, content={"error": f"Audio decode error: {e}"})
        os.remove(tmp_path)

        # 1. Transcribe (Speech-to-Text)
        user_text = cloner._transcribe(y)
        if not user_text or user_text.strip() == "":
            return JSONResponse(status_code=400, content={"error": "Could not understand audio. Please try speaking again."})

        # 2. Chat/LLM (Text-to-Text)
        ai_reply = conversation.respond(user_text, profile)

        # 3. Synthesise AI audio (Text-to-Speech)
        if target_file is not None:
            target_voice_bytes = await target_file.read()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                target_voice_tmp = tmp2.name
                tmp2.write(target_voice_bytes)
            y_out = cloner.clone_tts(ai_reply, profile=profile, target_voice_path=target_voice_tmp)
        else:
            y_out = cloner.clone_tts(ai_reply, profile=profile)
            
        y_watermarked = apply_audio_watermark(y_out, sr=cloner.sr)

        # Encode resulting audio
        out_buffer = io.BytesIO()
        sf.write(out_buffer, y_watermarked, cloner.sr, format='WAV')
        audio_b64 = base64.b64encode(out_buffer.getvalue()).decode('utf-8')
        
        latency = (time.time() - start_time) * 1000
        # Optional: log the interaction latency 
        evaluator.log_to_csv("Meeting Interaction", latency=latency)

        return {
            "status": "success",
            "user_text": user_text,
            "ai_text": ai_reply,
            "audio_data": audio_b64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Interaction failed: {str(e)}"})
    finally:
        if target_voice_tmp and os.path.exists(target_voice_tmp):
            os.remove(target_voice_tmp)

# ── 5. AI Conversation Chat ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    profile: str = "profile_1"


@app.post("/conversation/chat")
async def chat(
    body: Optional[ChatRequest] = Body(default=None),
    message: Optional[str] = Query(default=None, description="User message (legacy query form)"),
    profile: str = Query(default="profile_1"),
):
    """Generates a personality-based AI response for the conversation module.

    Accepts either a JSON body `{"message": ..., "profile": ...}` or the
    legacy query-string form so existing clients keep working.
    """
    # Resolve message + profile from body or query
    if body is not None and body.message:
        msg = body.message
        prof = body.profile or profile
    elif message:
        msg = message
        prof = profile
    else:
        return JSONResponse(status_code=400, content={"error": "Missing 'message'."})

    # Run the (potentially blocking) Gemini / offline call off the event loop
    start_time = time.time()
    reply = await asyncio.to_thread(conversation.respond, msg, prof)
    latency = (time.time() - start_time) * 1000
    evaluator.log_to_csv("Conversation", latency=latency)
    return {"reply": reply, "profile": prof}


# ── 6. CSV Report Export ─────────────────────────────────────────────
@app.get("/export/report")
async def export_report():
    """Downloads the session CSV containing all logged evaluation metrics."""
    report_path = "session_metrics.csv"
    if os.path.exists(report_path):
        filename = f"DSP_Project_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return FileResponse(
            path=report_path,
            filename=filename,
            media_type="text/csv"
        )
    return JSONResponse(status_code=404, content={"error": "No report data found. Run some processing first."})


@app.post("/process/swap-video")
async def swap_video(
    video:       UploadFile = File(...),
    profile:     str        = Form("profile_1"),
    target_file: Optional[UploadFile] = File(None),
    hd:          bool       = Form(False),
):
    """
    Process every frame of an uploaded video through the face-swap pipeline.
    Returns an MP4 with the faces swapped (audio preserved when ffmpeg is available).

    `hd=True` runs CodeFormer restoration on every frame (much slower; only
    advisable for short clips on this 4 GB GPU).
    """
    # Log active execution provider so it's visible per-request
    _eps = resolve_providers()
    _ep_name = (_eps[0][0] if isinstance(_eps[0], tuple) else _eps[0]) if _eps else "Unknown"
    print(f"[swap-video] Starting — EP: {_ep_name} | hd={hd}")

    # 1. Save uploaded video to a temp file
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(await video.read())
    tmp_in.close()

    # 2. Resolve target face image
    target_img = None
    if target_file is not None:
        target_arr = np.frombuffer(await target_file.read(), np.uint8)
        target_img = cv2.imdecode(target_arr, cv2.IMREAD_COLOR)
    if target_img is None:
        tpl_path = os.path.join(
            os.path.dirname(__file__), "dsp_models", "templates", f"{profile}.jpg"
        )
        target_img = cv2.imread(tpl_path)
        if target_img is None:
            os.unlink(tmp_in.name)
            return JSONResponse(status_code=400, content={"error": "Target face not found."})

    # 3. Open video and process frame by frame
    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        os.unlink(tmp_in.name)
        return JSONResponse(status_code=400, content={"error": "Cannot open video file."})

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_vid.close()
    writer = cv2.VideoWriter(
        tmp_vid.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    processed = 0
    _t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            swapped = await asyncio.to_thread(
                face_swapper.swap_with_target, frame, target_img, False, hd, False
            )
        except Exception:
            swapped = frame
        writer.write(swapped)
        processed += 1
        if processed % 10 == 0 or processed == 1:
            elapsed = time.time() - _t0
            fps_proc = processed / elapsed if elapsed > 0 else 0
            if total and fps_proc > 0:
                eta = (total - processed) / fps_proc
                pct = f"{100 * processed / total:.0f}%"
                print(f"[swap-video] {processed}/{total} frames ({pct}) | {fps_proc:.1f} fr/s | ETA {eta:.0f}s")
            else:
                print(f"[swap-video] {processed} frames | {fps_proc:.1f} fr/s")

    cap.release()
    writer.release()
    print(f"[swap-video] Done — {processed} frames at {fps:.1f} fps.")

    # 4. Mux original audio back using ffmpeg (best-effort)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    try:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_bin = "ffmpeg"

        proc = subprocess.run(
            [ffmpeg_bin, "-y",
             "-i", tmp_vid.name,
             "-i", tmp_in.name,
             "-map", "0:v:0",
             "-map", "1:a?",
             "-c:v", "libx264",
             "-preset", "fast",
             "-crf", "23",
             "-pix_fmt", "yuv420p",
             "-c:a", "aac",
             "-shortest",
             "-movflags", "+faststart",
             tmp_out.name],
            capture_output=True, timeout=300
        )
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg non-zero exit")
        os.unlink(tmp_vid.name)
        output_path = tmp_out.name
    except Exception:
        os.unlink(tmp_out.name)
        output_path = tmp_vid.name  # return video-only if ffmpeg unavailable

    os.unlink(tmp_in.name)

    def _cleanup(*_):
        try:
            os.unlink(output_path)
        except OSError:
            pass

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="swapped_video.mp4",
        background=BackgroundTask(_cleanup),
    )


# ── 5. Live Webcam Face Swap (WebSocket) ────────────────────────────
def _decode_data_url_jpeg(data: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG (optionally a 'data:image/...;base64,' URL) to BGR."""
    if not data:
        return None
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(data), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _stamp_live_watermark(frame: np.ndarray) -> np.ndarray:
    """Stamp a small parody/AI-generated label in the corner of a live frame."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    label = "AI GENERATED - PARODY"
    scale = max(0.4, w / 1280.0)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = 10, h - 12
    cv2.rectangle(frame, (x - 5, y - th - 8), (x + tw + 5, y + 6), (0, 0, 0), -1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 255), thickness, cv2.LINE_AA)
    return frame


@app.websocket("/ws/live-swap")
async def live_swap(ws: WebSocket):
    """
    Real-time webcam face swap over a WebSocket.

    Protocol (all messages are JSON text):
      1. Client → server, ONCE on connect, a config message:
           {"type": "config", "profile": "profile_1"}
         or with a custom target face:
           {"type": "config", "target_b64": "<base64 jpeg>"}
      2. Client → server, repeatedly, a frame message:
           {"type": "frame", "data": "<base64 jpeg of the webcam frame>"}
         Server → client replies with:
           {"type": "result", "data": "<base64 jpeg of the swapped frame>"}

    The target identity is detected ONCE (cached) so only the incoming webcam
    face is detected per frame.  No HD restoration is run (kept fast for the
    4 GB GPU); a parody watermark is stamped on every output frame.
    """
    await ws.accept()
    target_face = None

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            # ── Session configuration: resolve the target identity once ──
            if mtype == "config":
                target_img = None
                if msg.get("target_b64"):
                    target_img = _decode_data_url_jpeg(msg["target_b64"])
                if target_img is None:
                    profile = msg.get("profile", "profile_1")
                    tpl = os.path.join(
                        os.path.dirname(__file__), "dsp_models", "templates", f"{profile}.jpg"
                    )
                    target_img = cv2.imread(tpl)
                if target_img is None:
                    await ws.send_json({"type": "error", "message": "Target face not found."})
                    continue
                target_face = await asyncio.to_thread(
                    face_swapper.prepare_target_face, target_img
                )
                if target_face is None:
                    await ws.send_json({"type": "error", "message": "No face found in target image."})
                else:
                    await ws.send_json({"type": "ready"})
                continue

            # ── Per-frame swap ──────────────────────────────────────────
            if mtype == "frame":
                frame = _decode_data_url_jpeg(msg.get("data", ""))
                if frame is None:
                    continue
                if target_face is None:
                    out = frame  # not configured yet — echo unchanged
                else:
                    out = await asyncio.to_thread(face_swapper.swap_live, frame, target_face)
                    out = _stamp_live_watermark(out)
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    await ws.send_json({"type": "result", "data": b64})
                continue

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[live-swap] session ended: {type(e).__name__}: {e}")
        try:
            await ws.close()
        except Exception:
            pass


# ─────────────────────────── Entry Point ────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000,
                reload=True, reload_dirs=["backend"], reload_includes=["*.py"])
