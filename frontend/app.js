/**
 * app.js — Frontend Logic for AI Deepfake Interaction System
 * CENG 384 - Digital Signal Processing Project
 *
 * Sections:
 *  1. Config & State
 *  2. Tab Navigation
 *  3. API Health Check
 *  4. Voice Cloning Module
 *  5. Face Swap Module
 *  6. AI Conversation Module
 *  7. Meeting Simulation Module
 *  8. Evaluation / Metrics Module
 *  9. Waveform Canvas Visualiser
 * 10. Metrics Bar Chart (Canvas)
 * 11. Utility Helpers
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════
// 1. CONFIG & SHARED STATE
// ═══════════════════════════════════════════════════════════════════

const API_BASE = (
    location.protocol === 'file:' ||
    (location.hostname === '127.0.0.1' && location.port === '8000')
)
    ? 'http://127.0.0.1:8000'
    : `${location.protocol}//${location.host}`;

/** Holds the latest metric values — updated by every processing call. */
const sessionState = {
    latency: null,
    ssim: null,
    psnr: null,
    snr: null,
    mcd: null,
    log: [],          // array of session log row objects
};

// ═══════════════════════════════════════════════════════════════════
// 2. TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;

            tabBtns.forEach(b => {
                b.classList.toggle('active', b === btn);
                b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
            });

            tabPanels.forEach(panel => {
                const isTarget = panel.id === `panel-${target}`;
                panel.classList.toggle('active', isTarget);
            });

            // Refresh chart when navigating to metrics tab
            if (target === 'metrics') renderMetricsChart();
        });
    });
}

// ═══════════════════════════════════════════════════════════════════
// 3. API HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════

const apiDot = document.getElementById('api-status-dot');

async function checkApiHealth() {
    apiDot.className = 'status-dot checking';
    try {
        // AbortSignal.timeout is not available in older Safari/Edge
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 3000);
        let res;
        try {
            res = await fetch(`${API_BASE}/`, { signal: controller.signal });
        } finally {
            clearTimeout(timer);
        }
        if (res.ok) {
            apiDot.className = 'status-dot online';
            apiDot.title = 'Backend: Online';
        } else {
            throw new Error('non-200');
        }
    } catch {
        apiDot.className = 'status-dot offline';
        apiDot.title = 'Backend: Offline — start uvicorn first';
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. VOICE CLONING MODULE
// ═══════════════════════════════════════════════════════════════════

(function initVoiceModule() {
    // ── Elements ──
    const audioDropZone = document.getElementById('audio-drop-zone');
    const audioInput = document.getElementById('audio-input');
    const audioFilename = document.getElementById('audio-filename');
    const targetVoiceZone = document.getElementById('target-voice-drop-zone');
    const targetVoiceInput = document.getElementById('target-voice-input');
    const targetVoiceName = document.getElementById('target-voice-filename');
    const pitchSlider = document.getElementById('pitch-slider');
    const pitchValue = document.getElementById('pitch-value');
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    const emotionSelect = document.getElementById('emotion-select');
    const processVoiceBtn = document.getElementById('process-voice-btn');
    const processTtsBtn = document.getElementById('process-tts-btn');
    const ttsInput = document.getElementById('tts-input');
    const voiceStatus = document.getElementById('voice-status');
    const audioEmptyState = document.getElementById('audio-empty-state');
    const audioResultBlock = document.getElementById('audio-result-block');
    const audioPlayer = document.getElementById('audio-player');
    const voiceSnr = document.getElementById('voice-snr');
    const voiceMcd = document.getElementById('voice-mcd');
    const voiceLatency = document.getElementById('voice-latency');

    // ── Pitch slider label ──
    pitchSlider.addEventListener('input', () => {
        const v = parseFloat(pitchSlider.value);
        pitchValue.textContent = `${v >= 0 ? '+' : ''}${v} semitones`;
    });

    // ── Speed slider label ──
    if (speedSlider && speedValue) {
        speedSlider.addEventListener('input', () => {
            const s = parseFloat(speedSlider.value);
            speedValue.textContent = `${s.toFixed(1)}x`;
        });
    }

    // ── Drop zones ──
    setupDropZone(audioDropZone, audioInput, audioFilename, ['.wav', '.mp3', 'audio/']);
    setupDropZone(targetVoiceZone, targetVoiceInput, targetVoiceName, ['.wav', '.mp3', 'audio/']);

    // ── Process uploaded audio ──
    processVoiceBtn.addEventListener('click', async () => {
        const file = audioInput.files[0];
        if (!file) {
            flashError(voiceStatus, 'Please select a WAV or MP3 file first.');
            return;
        }
        setStatus(voiceStatus, 'processing', 'Processing…');
        setBtnLoading(processVoiceBtn, true);

        const formData = new FormData();
        formData.append('file', file);

        // Attach optional custom target voice
        const targetVoice = targetVoiceInput.files[0];
        if (targetVoice) {
            formData.append('target_file', targetVoice);
        }

        const pitch = parseFloat(pitchSlider.value);
        const speed = speedSlider ? parseFloat(speedSlider.value) : 1.0;
        const emotion = emotionSelect ? emotionSelect.value : 'default';
        const profile = document.getElementById('voice-profile-select').value;

        try {
            const res = await fetch(`${API_BASE}/process/voice?pitch_steps=${pitch}&profile=${profile}&speed_ratio=${speed}&emotion=${emotion}`, { method: 'POST', body: formData });
            const result = await handleResponse(res);
            showAudioResult(result);
        } catch (err) {
            setStatus(voiceStatus, 'error', `Error: ${err.message}`);
        } finally {
            setBtnLoading(processVoiceBtn, false);
        }
    });

    // ── Process TTS text ──
    processTtsBtn.addEventListener('click', async () => {
        const text = ttsInput.value.trim();
        if (!text) {
            flashError(voiceStatus, 'Please enter some text first.');
            return;
        }
        setStatus(voiceStatus, 'processing', 'Synthesising…');
        setBtnLoading(processTtsBtn, true);

        const profile = document.getElementById('voice-profile-select').value;
        const pitch = parseFloat(pitchSlider.value);
        const speed = speedSlider ? parseFloat(speedSlider.value) : 1.0;
        const emotion = emotionSelect ? emotionSelect.value : 'default';

        // Build FormData for TTS (to support optional target_file upload)
        const formData = new FormData();
        const targetVoice = targetVoiceInput.files[0];
        if (targetVoice) {
            formData.append('target_file', targetVoice);
        }

        try {
            const url = `${API_BASE}/process/tts?text=${encodeURIComponent(text)}&profile=${profile}&pitch_steps=${pitch}&speed_ratio=${speed}&emotion=${emotion}`;
            const res = await fetch(url, { method: 'POST', body: formData });
            const result = await handleResponse(res);
            showAudioResult(result);
        } catch (err) {
            setStatus(voiceStatus, 'error', `Error: ${err.message}`);
        } finally {
            setBtnLoading(processTtsBtn, false);
        }
    });

    function showAudioResult(result) {
        // Decode base64 WAV and play
        const blob = b64toBlob(result.audio_data, 'audio/wav');
        audioPlayer.src = URL.createObjectURL(blob);

        // Show result panel
        audioEmptyState.classList.add('hidden');
        audioResultBlock.classList.remove('hidden');

        // Update mini metrics
        voiceSnr.textContent = `${result.snr} dB`;
        voiceMcd.textContent = result.mcd;
        voiceLatency.textContent = `${result.latency_ms} ms`;

        // Handle transcript if available
        const transcriptBlock = document.getElementById('voice-transcript-block');
        const transcriptText = document.getElementById('voice-transcript-text');
        if (transcriptBlock && transcriptText) {
            if (result.transcript) {
                transcriptText.textContent = result.transcript;
                transcriptBlock.classList.remove('hidden');
            } else {
                transcriptBlock.classList.add('hidden');
            }
        }

        // Update shared state and global metrics panel
        updateSharedMetrics({ snr: result.snr, mcd: result.mcd, latency_ms: result.latency_ms }, 'Audio');
        setStatus(voiceStatus, 'success', '✓ Complete');

        // Draw waveform
        drawWaveformFromBlob(blob);
    }
})();

// ═══════════════════════════════════════════════════════════════════
// 5. FACE SWAP MODULE
// ═══════════════════════════════════════════════════════════════════

(function initFaceModule() {
    const mediaDropZone = document.getElementById('media-drop-zone');
    const mediaInput = document.getElementById('media-input');
    const mediaFilename = document.getElementById('media-filename');
    const targetFaceZone = document.getElementById('target-face-drop-zone');
    const targetFaceInput = document.getElementById('target-face-input');
    const targetFaceName = document.getElementById('target-face-filename');
    const processFaceBtn = document.getElementById('process-face-btn');
    const faceStatus = document.getElementById('face-status');
    const facePlaceholder = document.getElementById('face-placeholder');
    const originalPreviewOnly = document.getElementById('original-preview-only');
    const sliderInteractive = document.getElementById('slider-interactive');
    const sliderOriginal = document.getElementById('slider-original');
    const sliderProcessed = document.getElementById('slider-processed');
    const sliderRange = document.getElementById('face-slider-range');
    const sliderOverlay = document.getElementById('slider-overlay');
    const sliderLine = document.getElementById('slider-handle-line');
    const sliderBtn = document.getElementById('slider-handle-button');
    const faceProfileSelect = document.getElementById('face-profile-select');

    if (sliderRange) {
        sliderRange.addEventListener('input', (e) => {
            const val = e.target.value;
            sliderOverlay.style.clipPath = `polygon(0 0, ${val}% 0, ${val}% 100%, 0 100%)`;
            sliderLine.style.left = `${val}%`;
            sliderBtn.style.left = `${val}%`;
        });
    }
    const faceSSIM = document.getElementById('face-ssim');
    const facePSNR = document.getElementById('face-psnr');
    const faceLatency = document.getElementById('face-latency');

    setupDropZone(mediaDropZone, mediaInput, mediaFilename, ['image/']);
    setupDropZone(targetFaceZone, targetFaceInput, targetFaceName, ['image/']);

    // Show original preview immediately on file select
    mediaInput.addEventListener('change', () => {
        const file = mediaInput.files[0];
        if (!file) return;
        if (file.size > 10 * 1024 * 1024) {
            alert('File too large — please select an image under 10 MB.');
            mediaInput.value = '';
            return;
        }
        const objUrl = URL.createObjectURL(file);
        originalPreviewOnly.src = objUrl;
        sliderOriginal.src = objUrl;
        
        facePlaceholder.style.display = 'none';
        originalPreviewOnly.style.display = 'block';
        originalPreviewOnly.classList.remove('hidden');
        sliderInteractive.classList.add('hidden');
    });

    processFaceBtn.addEventListener('click', async () => {
        const file = mediaInput.files[0];
        if (!file) {
            flashError(faceStatus, 'Please select an image first.');
            return;
        }
        setStatus(faceStatus, 'processing', 'Processing…');
        setBtnLoading(processFaceBtn, true);

        const formData = new FormData();
        formData.append('file', file);

        // Attach optional custom target face
        const targetFace = targetFaceInput.files[0];
        if (targetFace) {
            formData.append('target_file', targetFace);
        }

        const profile = faceProfileSelect.value;
        const traceFaces = document.getElementById('trace-faces-checkbox').checked;

        try {
            const res = await fetch(`${API_BASE}/process/frame?profile=${profile}&trace_faces=${traceFaces}`, { method: 'POST', body: formData });
            const result = await handleResponse(res);

            // Display processed image
            sliderProcessed.src = `data:image/jpeg;base64,${result.image_data}`;
            
            // Switch to slider view
            originalPreviewOnly.style.display = 'none';
            sliderInteractive.classList.remove('hidden');
            
            // Reset slider position
            sliderRange.value = 50;
            sliderOverlay.style.clipPath = `polygon(0 0, 50% 0, 50% 100%, 0 100%)`;
            sliderLine.style.left = `50%`;
            sliderBtn.style.left = `50%`;

            // Metrics
            faceSSIM.textContent = result.ssim;
            facePSNR.textContent = `${result.psnr} dB`;
            faceLatency.textContent = `${result.latency_ms} ms`;

            updateSharedMetrics({ ssim: result.ssim, psnr: result.psnr, latency_ms: result.latency_ms }, 'Video/Image');
            setStatus(faceStatus, 'success', '✓ Complete');

            // Handle tracing evaluation image
            const tracingContainer = document.getElementById('tracing-eval-container');
            if (result.tracing_image_data) {
                document.getElementById('tracing-result-img').src = `data:image/jpeg;base64,${result.tracing_image_data}`;
                tracingContainer.classList.remove('hidden');
                // Auto-switch to Evaluation tab so the user sees the geometry immediately
                const metricsTabBtn = document.querySelector('.tab-btn[data-tab="metrics"]');
                if (metricsTabBtn) metricsTabBtn.click();
            } else if (tracingContainer) {
                tracingContainer.classList.add('hidden');
            }
        } catch (err) {
            setStatus(faceStatus, 'error', `Error: ${err.message}`);
        } finally {
            setBtnLoading(processFaceBtn, false);
        }
    });
})();

// ═══════════════════════════════════════════════════════════════════
// 5b. VIDEO DEEPFAKE MODULE (frame-by-frame)
// ═══════════════════════════════════════════════════════════════════

(function initVideoModule() {
    const videoDropZone   = document.getElementById('video-drop-zone');
    const videoInput      = document.getElementById('video-input');
    const videoFilename   = document.getElementById('video-filename');
    const targetDropZone  = document.getElementById('video-target-drop-zone');
    const targetInput     = document.getElementById('video-target-input');
    const targetFilename  = document.getElementById('video-target-filename');
    const processBtn      = document.getElementById('process-video-btn');
    const statusBadge     = document.getElementById('video-status');
    const placeholder     = document.getElementById('video-placeholder');
    const resultVideo     = document.getElementById('video-result');
    const downloadLink    = document.getElementById('video-download');
    const profileSelect   = document.getElementById('video-profile-select');
    const hdCheckbox      = document.getElementById('video-hd-checkbox');

    if (!videoInput) return;  // panel not present

    setupDropZone(videoDropZone, videoInput, videoFilename, ['video/']);
    setupDropZone(targetDropZone, targetInput, targetFilename, ['image/']);

    let lastObjectUrl = null;

    processBtn.addEventListener('click', async () => {
        const file = videoInput.files[0];
        if (!file) {
            flashError(statusBadge, 'Please select a video first.');
            return;
        }
        if (file.size > 100 * 1024 * 1024) {
            flashError(statusBadge, 'Video too large — keep it under 100 MB.');
            return;
        }

        setStatus(statusBadge, 'processing', 'Processing frames… this can take a while');
        setBtnLoading(processBtn, true);

        const formData = new FormData();
        formData.append('video', file);
        formData.append('profile', profileSelect.value);
        formData.append('hd', hdCheckbox.checked ? 'true' : 'false');
        const targetFace = targetInput.files[0];
        if (targetFace) formData.append('target_file', targetFace);

        try {
            const res = await fetch(`${API_BASE}/process/swap-video`, {
                method: 'POST', body: formData,
            });
            if (!res.ok) {
                let msg = `HTTP ${res.status}`;
                try { const j = await res.json(); if (j.error) msg = j.error; } catch (_) {}
                throw new Error(msg);
            }

            const blob = await res.blob();
            if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
            lastObjectUrl = URL.createObjectURL(blob);

            placeholder.style.display = 'none';
            resultVideo.src = lastObjectUrl;
            resultVideo.classList.remove('hidden');
            downloadLink.href = lastObjectUrl;
            downloadLink.classList.remove('hidden');

            setStatus(statusBadge, 'success', '✓ Complete');
        } catch (err) {
            setStatus(statusBadge, 'error', `Error: ${err.message}`);
        } finally {
            setBtnLoading(processBtn, false);
        }
    });
})();

// ═══════════════════════════════════════════════════════════════════
// 5b. LIVE WEBCAM SWAP MODULE (real-time, WebSocket)
// ═══════════════════════════════════════════════════════════════════

(function initLiveSwapModule() {
    const webcam        = document.getElementById('live-webcam');
    const canvas        = document.getElementById('live-canvas');
    const placeholder   = document.getElementById('live-placeholder');
    const startBtn      = document.getElementById('live-start-btn');
    const stopBtn       = document.getElementById('live-stop-btn');
    const statusBadge   = document.getElementById('live-status');
    const profileSelect = document.getElementById('live-profile-select');
    const targetDrop    = document.getElementById('live-target-drop-zone');
    const targetInput   = document.getElementById('live-target-input');
    const targetName    = document.getElementById('live-target-filename');
    const fpsWrap       = document.getElementById('live-fps');
    const fpsValue      = document.getElementById('live-fps-value');

    if (!webcam) return;  // panel not present

    setupDropZone(targetDrop, targetInput, targetName, ['image/']);

    const WS_URL = API_BASE.replace(/^http/, 'ws') + '/ws/live-swap';
    const SEND_W = 480;   // downscale frames sent to the server (speed)

    let ws = null;
    let stream = null;
    let running = false;
    let sending = false;          // one in-flight frame at a time
    const grab = document.createElement('canvas');  // offscreen capture
    const ctx = canvas.getContext('2d');
    let frameCount = 0, fpsTimer = 0;

    function fileToDataUrl(file) {
        return new Promise((resolve, reject) => {
            const fr = new FileReader();
            fr.onload = () => resolve(fr.result);
            fr.onerror = reject;
            fr.readAsDataURL(file);
        });
    }

    function sendFrame() {
        if (!running || sending || !ws || ws.readyState !== WebSocket.OPEN) return;
        if (!webcam.videoWidth) return;
        const ratio = webcam.videoHeight / webcam.videoWidth;
        const w = SEND_W, h = Math.round(SEND_W * ratio);
        grab.width = w; grab.height = h;
        const gctx = grab.getContext('2d');
        gctx.drawImage(webcam, 0, 0, w, h);
        const dataUrl = grab.toDataURL('image/jpeg', 0.7);
        sending = true;
        ws.send(JSON.stringify({ type: 'frame', data: dataUrl }));
    }

    function drawResult(b64) {
        const img = new Image();
        img.onload = () => {
            if (canvas.width !== img.width || canvas.height !== img.height) {
                canvas.width = img.width;
                canvas.height = img.height;
            }
            ctx.drawImage(img, 0, 0);
            // FPS counter
            frameCount++;
            const now = performance.now();
            if (now - fpsTimer >= 1000) {
                fpsValue.textContent = frameCount;
                frameCount = 0;
                fpsTimer = now;
            }
        };
        img.src = 'data:image/jpeg;base64,' + b64;
        sending = false;
        if (running) requestAnimationFrame(sendFrame);
    }

    async function start() {
        try {
            setStatus(statusBadge, 'processing', 'Requesting camera…');
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640 }, audio: false });
            webcam.srcObject = stream;
            await webcam.play();
        } catch (err) {
            setStatus(statusBadge, 'error', `Camera blocked: ${err.message}`);
            return;
        }

        setStatus(statusBadge, 'processing', 'Connecting…');
        ws = new WebSocket(WS_URL);

        ws.onopen = async () => {
            const cfg = { type: 'config', profile: profileSelect.value };
            const tf = targetInput.files[0];
            if (tf) cfg.target_b64 = await fileToDataUrl(tf);
            ws.send(JSON.stringify(cfg));
        };

        ws.onmessage = (ev) => {
            let msg;
            try { msg = JSON.parse(ev.data); } catch (_) { return; }
            if (msg.type === 'ready') {
                running = true;
                placeholder.style.display = 'none';
                canvas.classList.remove('hidden');
                fpsWrap.classList.remove('hidden');
                startBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
                fpsTimer = performance.now();
                setStatus(statusBadge, 'success', '● Live');
                requestAnimationFrame(sendFrame);
            } else if (msg.type === 'result') {
                drawResult(msg.data);
            } else if (msg.type === 'error') {
                setStatus(statusBadge, 'error', msg.message || 'Server error');
            }
        };

        ws.onerror = () => setStatus(statusBadge, 'error', 'WebSocket error');
        ws.onclose = () => { if (running) stop(); };
    }

    function stop() {
        running = false;
        sending = false;
        if (ws) { try { ws.close(); } catch (_) {} ws = null; }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        webcam.srcObject = null;
        canvas.classList.add('hidden');
        fpsWrap.classList.add('hidden');
        placeholder.style.display = 'flex';
        startBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        setStatus(statusBadge, 'idle', '● Idle');
    }

    startBtn.addEventListener('click', start);
    stopBtn.addEventListener('click', stop);
    window.addEventListener('beforeunload', () => { if (running) stop(); });
})();

// ═══════════════════════════════════════════════════════════════════
// 6. AI CONVERSATION MODULE
// ═══════════════════════════════════════════════════════════════════

(function initChatModule() {
    const chatLog = document.getElementById('chat-log');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatProfileSel = document.getElementById('chat-profile-select');

    function appendBubble(text, role, profileName) {
        const div = document.createElement('div');
        div.classList.add('chat-bubble', role);

        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        div.innerHTML = `
            ${escHtml(text)}
            <div class="bubble-meta">${role === 'ai' ? profileName + ' (AI) · ' : 'You · '}${time}</div>
        `;
        chatLog.appendChild(div);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function sendMessage() {
        const msg = chatInput.value.trim();
        if (!msg) return;

        const profile = chatProfileSel.value;
        chatInput.value = '';
        appendBubble(msg, 'user', '');
        setBtnLoading(chatSendBtn, true);

        try {
            const url = `${API_BASE}/conversation/chat`;
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, profile }),
            });
            const data = await handleResponse(res);

            const profileLabel = profile === 'profile_1' ? 'Public Figure 1' : 'Public Figure 2';
            appendBubble(data.reply, 'ai', profileLabel);
        } catch (err) {
            appendBubble(`⚠ Backend error: ${err.message}`, 'ai', 'System');
        } finally {
            setBtnLoading(chatSendBtn, false);
            chatInput.focus();
        }
    }

    chatSendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
})();

// ═══════════════════════════════════════════════════════════════════
// 7. MEETING SIMULATION MODULE
// ═══════════════════════════════════════════════════════════════════

(function initMeetingModule() {
    const meetingTargetVoiceGroup = document.getElementById('meeting-target-voice-group');
    const meetingTargetVoiceZone = document.getElementById('meeting-target-voice-drop-zone');
    const meetingTargetVoiceInput = document.getElementById('meeting-target-voice-input');
    const meetingTargetVoiceName = document.getElementById('meeting-target-voice-filename');
    const startMeetingBtn = document.getElementById('start-meeting-btn');
    const scenarioSelect = document.getElementById('scenario-select');
    const meetingProfileSelect = document.getElementById('meeting-profile-select');
    const readAloudToggle = document.getElementById('meeting-read-aloud');
    const scriptBox = document.getElementById('script-box');
    const nextLineBtn = document.getElementById('next-line-btn');
    const prevLineBtn = document.getElementById('prev-line-btn');
    const webcamBtn = document.getElementById('webcam-btn');
    const recordBtn = document.getElementById('record-btn');
    const localVideo = document.getElementById('local-video');
    const camPlaceholder = document.getElementById('cam-placeholder');
    const speakingRing = document.getElementById('speaking-ring');
    const progressBar = document.getElementById('progress-bar');
    const progressBarWrap = document.getElementById('progress-bar-wrap');
    const aiAvatarImg = document.getElementById('ai-avatar-img');
    const aiAvatarFallback = document.getElementById('ai-avatar-fallback');
    const transcript = document.getElementById('meeting-transcript');
    const transcriptEmpty = document.getElementById('transcript-empty');
    const transcriptCount = document.getElementById('transcript-count');

    let currentScript = null;
    let lineIndex = 0;
    let webcamActive = false;
    let allScenarios = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let transcriptLineCount = 0;
    let isSpeaking = false;

    setupDropZone(meetingTargetVoiceZone, meetingTargetVoiceInput, meetingTargetVoiceName, ['.wav', '.mp3', 'audio/']);

    // Toggle custom voice input based on profile
    meetingProfileSelect.addEventListener('change', () => {
        if (meetingProfileSelect.value === 'free_ai') {
            meetingTargetVoiceGroup.classList.remove('hidden');
        } else {
            meetingTargetVoiceGroup.classList.add('hidden');
            // Clear out any uploaded file if switched away from free_ai
            meetingTargetVoiceInput.value = '';
            meetingTargetVoiceName.textContent = '';
            meetingTargetVoiceName.classList.add('hidden');
            meetingTargetVoiceZone.classList.remove('hidden');
        }
    });

    // ── Audio Reactivity Setup ──
    let audioCtx = null;
    let analyser = null;
    let dataArray = null;

    function playAudioWithReactivity(audioObj) {
        if (!audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContext();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            dataArray = new Uint8Array(analyser.frequencyBinCount);
        }

        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const source = audioCtx.createMediaElementSource(audioObj);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);

        let animationId;
        const canvas = document.getElementById('ai-visualizer');
        const ctx = canvas ? canvas.getContext('2d') : null;

        function animate() {
            if (!ctx) return;
            if (audioObj.paused || audioObj.ended) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'rgba(157, 78, 221, 0.2)';
                ctx.fillRect(0, canvas.height / 2 - 2, canvas.width, 4);
                return;
            }

            animationId = requestAnimationFrame(animate);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / dataArray.length) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < dataArray.length; i++) {
                barHeight = dataArray[i] / 2;
                ctx.fillStyle = `rgb(${dataArray[i] + 100}, 50, 221)`;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        }

        audioObj.addEventListener('play', () => {
            if (speakingRing) speakingRing.classList.remove('hidden');
            animate();
        });

        audioObj.addEventListener('ended', () => {
            if (speakingRing) speakingRing.classList.add('hidden');
            cancelAnimationFrame(animationId);
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'rgba(157, 78, 221, 0.2)';
                ctx.fillRect(0, canvas.height / 2 - 2, canvas.width, 4);
            }
        });

        audioObj.addEventListener('error', () => {
            if (speakingRing) speakingRing.classList.add('hidden');
            cancelAnimationFrame(animationId);
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        });

        return audioObj.play();
    }

    // Load scenarios from backend on init
    async function loadScenarios() {
        try {
            const res = await fetch(`${API_BASE}/simulation/scenarios`);
            allScenarios = await res.json();
        } catch {
            allScenarios = {
                comedy_interview: { title: 'Comedy Interview', script: [{ speaker: 'Host', text: 'Hello!' }, { speaker: 'AI', text: 'Hello back!' }] }
            };
        }
    }

    // Removed loadProfileAvatar as the user requested an audio visualizer instead.

    // Add a line to the transcript panel
    function addTranscriptLine(speaker, text, role) {
        if (transcriptEmpty) transcriptEmpty.classList.add('hidden');
        const div = document.createElement('div');
        div.className = `transcript-line ${role}`;
        div.innerHTML = `
            <div class="transcript-speaker">${escHtml(speaker)}</div>
            <div class="transcript-text">${escHtml(text)}</div>
        `;
        transcript.appendChild(div);
        transcript.scrollTop = transcript.scrollHeight;
        transcriptLineCount++;
        transcriptCount.textContent = `${transcriptLineCount} line${transcriptLineCount !== 1 ? 's' : ''}`;
    }

    // Clear transcript
    function clearTranscript() {
        transcript.innerHTML = '';
        if (transcriptEmpty) {
            transcriptEmpty.classList.remove('hidden');
            transcript.appendChild(transcriptEmpty);
        }
        transcriptLineCount = 0;
        transcriptCount.textContent = '0 lines';
    }

    // Speak an AI line via TTS
    async function speakLine(text) {
        if (!readAloudToggle.checked || isSpeaking) return;
        isSpeaking = true;
        if (speakingRing) speakingRing.classList.remove('hidden');

        // Show TTS loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'tts-loading';
        loadingDiv.innerHTML = '<span class="spinner"></span> Synthesising voice…';
        scriptBox.appendChild(loadingDiv);

        try {
            const profile = meetingProfileSelect.value;
            const url = `${API_BASE}/process/tts?text=${encodeURIComponent(text)}&profile=${profile}&pitch_steps=0`;
            
            const formData = new FormData();
            const targetVoice = meetingTargetVoiceInput.files[0];
            if (targetVoice) {
                formData.append('target_file', targetVoice);
            }
            
            const res = await fetch(url, { method: 'POST', body: formData });
            const data = await handleResponse(res);

            // Remove loading indicator
            if (loadingDiv.parentNode) loadingDiv.remove();

            // Play the audio
            const blob = b64toBlob(data.audio_data, 'audio/wav');
            const audioObj = new Audio(URL.createObjectURL(blob));

            audioObj.addEventListener('ended', () => { isSpeaking = false; });
            audioObj.addEventListener('error', () => { isSpeaking = false; });

            await playAudioWithReactivity(audioObj);
        } catch (e) {
            console.error('[Meeting TTS] Error:', e);
            if (loadingDiv.parentNode) loadingDiv.remove();
            if (speakingRing) speakingRing.classList.add('hidden');
            isSpeaking = false;
        }
    }

    // ── Start session ────────────────────────────────────────────
    startMeetingBtn.addEventListener('click', async () => {
        if (!allScenarios) await loadScenarios();

        // Clear previous transcript
        clearTranscript();

        const key = scenarioSelect.value;
        if (key === 'dynamic') {
            currentScript = null;
            nextLineBtn.classList.add('hidden');
            prevLineBtn.classList.add('hidden');
            progressBarWrap.classList.add('hidden');
            recordBtn.classList.remove('hidden');
            scriptBox.innerHTML = `<em>Dynamic Voice Chat ready. Toggle your camera (grants mic access) then <strong>HOLD</strong> the mic button to speak.</em>`;
            return;
        } else {
            nextLineBtn.classList.remove('hidden');
            prevLineBtn.classList.remove('hidden');
            progressBarWrap.classList.remove('hidden');
            recordBtn.classList.add('hidden');
        }

        const scenario = allScenarios[key];
        if (!scenario) return;

        currentScript = scenario.script;
        lineIndex = 0;
        nextLineBtn.disabled = false;
        prevLineBtn.disabled = false;
        displayLine();
    });

    // ── Script navigation ────────────────────────────────────────
    nextLineBtn.addEventListener('click', () => {
        if (!currentScript || isSpeaking) return;
        lineIndex = Math.min(lineIndex + 1, currentScript.length - 1);
        displayLine();
    });

    prevLineBtn.addEventListener('click', () => {
        if (!currentScript) return;
        lineIndex = Math.max(lineIndex - 1, 0);
        displayLine();
    });

    function displayLine() {
        if (!currentScript) return;
        const line = currentScript[lineIndex];

        // Update script box
        scriptBox.innerHTML = `<strong style="color:var(--purple)">${escHtml(line.speaker)}:</strong>  ${escHtml(line.text)}`;

        // Update progress bar
        const pct = ((lineIndex + 1) / currentScript.length) * 100;
        progressBar.style.width = `${pct}%`;

        // Add to transcript
        const role = line.speaker === 'AI' ? 'ai' : 'host';
        addTranscriptLine(line.speaker, line.text, role);

        // Auto-speak AI lines
        if (line.speaker === 'AI') {
            speakLine(line.text);
        }

        prevLineBtn.disabled = lineIndex === 0;
        nextLineBtn.disabled = lineIndex === currentScript.length - 1;
    }

    // ── Dynamic voice chat ───────────────────────────────────────
    recordBtn.addEventListener('mousedown', () => {
        if (!webcamActive || !localVideo.srcObject) {
            alert('Please toggle the camera to allow microphone access before recording.');
            return;
        }
        try {
            const stream = localVideo.srcObject;
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener('stop', async () => {
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                try {
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const actx = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await actx.decodeAudioData(arrayBuffer);

                    const numOfChan = audioBuffer.numberOfChannels;
                    const length = audioBuffer.length * numOfChan * 2 + 44;
                    const buffer = new ArrayBuffer(length);
                    const view = new DataView(buffer);
                    const channels = [];
                    let sample, offset = 0, pos = 0;

                    function setUint16(data) { view.setUint16(pos, data, true); pos += 2; }
                    function setUint32(data) { view.setUint32(pos, data, true); pos += 4; }

                    setUint32(0x46464952); setUint32(length - 8); setUint32(0x45564157);
                    setUint32(0x20746d66); setUint32(16); setUint16(1); setUint16(numOfChan);
                    setUint32(audioBuffer.sampleRate); setUint32(audioBuffer.sampleRate * 2 * numOfChan);
                    setUint16(numOfChan * 2); setUint16(16); setUint32(0x61746164); setUint32(length - pos - 4);

                    for (let i = 0; i < numOfChan; i++) channels.push(audioBuffer.getChannelData(i));

                    while (pos < length) {
                        for (let i = 0; i < numOfChan; i++) {
                            sample = Math.max(-1, Math.min(1, channels[i][offset]));
                            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
                            view.setInt16(pos, sample, true);
                            pos += 2;
                        }
                        offset++;
                    }

                    const wavBlob = new Blob([buffer], { type: "audio/wav" });
                    await sendDynamicVoice(wavBlob);
                } catch (e) {
                    console.error("WAV conversion error:", e);
                    await sendDynamicVoice(audioBlob); // fallback
                }
            });

            // Recording indicator
            recordBtn.innerHTML = '<span class="recording-indicator"><span class="recording-dot"></span> Recording…</span>';
            recordBtn.style.transform = 'scale(0.95)';
        } catch (e) {
            alert('Error starting recorder: ' + e.message);
        }
    });

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            recordBtn.style.transform = 'scale(1)';
            recordBtn.innerHTML = '<span class="spinner"></span> Processing…';
            recordBtn.disabled = true;
        }
    };

    recordBtn.addEventListener('mouseup', stopRecording);
    recordBtn.addEventListener('mouseleave', stopRecording);
    // Touch support for mobile/tablet
    recordBtn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        recordBtn.dispatchEvent(new MouseEvent('mousedown'));
    }, { passive: false });
    recordBtn.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopRecording();
    }, { passive: false });
    recordBtn.addEventListener('touchcancel', stopRecording);

    async function sendDynamicVoice(audioBlob) {
        try {
            const formData = new FormData();
            const filename = audioBlob.type === 'audio/wav' ? 'voice.wav' : 'voice.webm';
            formData.append('file', audioBlob, filename);
            const profile = meetingProfileSelect.value;
            
            const targetVoice = meetingTargetVoiceInput.files[0];
            if (targetVoice) {
                formData.append('target_file', targetVoice);
            }

            const res = await fetch(`${API_BASE}/simulation/interact?profile=${profile}`, {
                method: 'POST',
                body: formData
            });
            const data = await handleResponse(res);

            // Update script box with conversation
            scriptBox.innerHTML = `
                <div style="margin-bottom:8px; opacity: 0.8"><strong>You:</strong> ${escHtml(data.user_text)}</div>
                <div><strong style="color:var(--purple)">AI:</strong> ${escHtml(data.ai_text)}</div>
            `;

            // Add both lines to transcript
            addTranscriptLine('You', data.user_text, 'user');
            addTranscriptLine('AI', data.ai_text, 'ai');

            // Play AI response audio
            const blob = b64toBlob(data.audio_data, 'audio/wav');
            const audioObj = new Audio(URL.createObjectURL(blob));

            playAudioWithReactivity(audioObj).catch(e => console.error("Playback error:", e));

        } catch (err) {
            scriptBox.innerHTML = `<em>Failed to process interaction: ${escHtml(err.message)}</em>`;
        } finally {
            recordBtn.innerHTML = '&#127908; Hold to Talk';
            recordBtn.disabled = false;
        }
    }

    // ── Webcam toggle ────────────────────────────────────────────
    webcamBtn.addEventListener('click', async () => {
        if (webcamActive) {
            const stream = localVideo.srcObject;
            if (stream) stream.getTracks().forEach(t => t.stop());
            localVideo.srcObject = null;
            webcamActive = false;
            camPlaceholder.classList.remove('hidden');
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                localVideo.srcObject = stream;
                webcamActive = true;
                camPlaceholder.classList.add('hidden');
            } catch (err) {
                alert('Camera/Mic access denied or unavailable: ' + err.message);
                camPlaceholder.classList.remove('hidden');
            }
        }
    });

    // ── Export Transcript ──────────────────────────────────────────
    const exportTranscriptBtn = document.getElementById('export-transcript-btn');
    if (exportTranscriptBtn) {
        exportTranscriptBtn.addEventListener('click', () => {
            if (transcriptLineCount === 0) {
                alert("The transcript is empty.");
                return;
            }

            let text = "=== Meeting Simulation Transcript ===\n\n";
            const lines = transcript.querySelectorAll('.transcript-line');
            lines.forEach(line => {
                const speaker = line.querySelector('.transcript-speaker').textContent;
                const content = line.querySelector('.transcript-text').textContent;
                text += `${speaker}: ${content}\n`;
            });

            const blob = new Blob([text], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Meeting_Transcript_${new Date().getTime()}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // Pre-load scenarios on page load
    loadScenarios();
})();

// ═══════════════════════════════════════════════════════════════════
// 8. EVALUATION / METRICS MODULE
// ═══════════════════════════════════════════════════════════════════

(function initMetricsModule() {
    const exportBtn = document.getElementById('export-btn');

    exportBtn.addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE}/export/report`);
            if (!res.ok) {
                const err = await res.json();
                alert(err.error || 'Export failed — run some processing tasks first.');
                return;
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `DSP_Deepfake_Report_${new Date().toISOString().slice(0, 10)}.csv`;
            document.body.appendChild(a);
            a.click();
            URL.revokeObjectURL(url);
            a.remove();
            // Show metrics
            updateMetrics(result);
            updateChart();
        } catch (err) {
            alert('Export error: ' + err.message);
        }
    });
})();

/** Called by voice / face modules after each successful operation. */
function updateSharedMetrics(data, type) {
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // Update top-level metric cards
    if (data.latency_ms != null) animateMetric('metric-latency', `${data.latency_ms} ms`);
    if (data.ssim != null) animateMetric('metric-ssim', data.ssim);
    if (data.psnr != null) animateMetric('metric-psnr', `${data.psnr} dB`);
    if (data.snr != null) animateMetric('metric-snr', `${data.snr} dB`);
    if (data.mcd != null) animateMetric('metric-mcd', data.mcd);

    // Persist values
    if (data.latency_ms != null) sessionState.latency = data.latency_ms;
    if (data.ssim != null) sessionState.ssim = data.ssim;
    if (data.psnr != null) sessionState.psnr = data.psnr;
    if (data.snr != null) sessionState.snr = data.snr;
    if (data.mcd != null) sessionState.mcd = data.mcd;

    // Session log table row
    const row = {
        time: now,
        type,
        latency: data.latency_ms ?? '—',
        ssim: data.ssim ?? '—',
        psnr: data.psnr ?? '—',
        snr: data.snr ?? '—',
        mcd: data.mcd ?? '—',
    };
    sessionState.log.push(row);
    appendTableRow(row);
    renderMetricsChart();
}

function animateMetric(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value;
    el.classList.remove('updated');
    void el.offsetWidth; // reflow to re-trigger animation
    el.classList.add('updated');
    setTimeout(() => el.classList.remove('updated'), 600);
}

function appendTableRow(row) {
    const tbody = document.getElementById('session-log-body');
    // Remove the empty-state row if present (it's a single <tr> with one <td.empty-row>)
    const emptyTd = tbody.querySelector('td.empty-row');
    if (emptyTd) {
        const parentTr = emptyTd.parentElement;
        if (parentTr) parentTr.remove();
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td>${escHtml(row.time)}</td>
        <td>${escHtml(row.type)}</td>
        <td>${row.latency}</td>
        <td>${row.ssim}</td>
        <td>${row.psnr}</td>
        <td>${row.snr}</td>
        <td>${row.mcd}</td>
    `;
    tbody.appendChild(tr);
}

// ═══════════════════════════════════════════════════════════════════
// 9. WAVEFORM CANVAS VISUALISER
// ═══════════════════════════════════════════════════════════════════

async function drawWaveformFromBlob(blob) {
    const canvas = document.getElementById('waveform-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    try {
        const arrayBuf = await blob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
        const data = audioBuf.getChannelData(0);
        const step = Math.floor(data.length / W);

        ctx.clearRect(0, 0, W, H);

        // Gradient fill
        const grad = ctx.createLinearGradient(0, 0, W, 0);
        grad.addColorStop(0, '#6c63ff');
        grad.addColorStop(0.5, '#3ecfcf');
        grad.addColorStop(1, '#f857a6');
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;

        ctx.beginPath();
        for (let i = 0; i < W; i++) {
            const sample = data[i * step] ?? 0;
            const y = (H / 2) + (sample * H * 0.45);
            i === 0 ? ctx.moveTo(i, y) : ctx.lineTo(i, y);
        }
        ctx.stroke();
        audioCtx.close();
    } catch {
        // Silently skip waveform if browser AudioContext is unavailable
    }
}

// ═══════════════════════════════════════════════════════════════════
// 10. METRICS BAR CHART (Canvas)
// ═══════════════════════════════════════════════════════════════════

function renderMetricsChart() {
    const canvas = document.getElementById('metrics-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // Normalise metrics to 0–100 scale for display
    const bars = [
        { label: 'SSIM', raw: sessionState.ssim, norm: sessionState.ssim != null ? sessionState.ssim * 100 : 0, color: '#6c63ff' },
        { label: 'PSNR', raw: sessionState.psnr, norm: sessionState.psnr != null ? Math.min(sessionState.psnr, 60) / 60 * 100 : 0, color: '#3ecfcf' },
        { label: 'SNR', raw: sessionState.snr, norm: sessionState.snr != null ? Math.min(Math.max(sessionState.snr, 0), 60) / 60 * 100 : 0, color: '#f857a6' },
        { label: 'MCD', raw: sessionState.mcd, norm: sessionState.mcd != null ? Math.max(0, 100 - sessionState.mcd * 5) : 0, color: '#fb923c' },
        { label: 'Speed', raw: sessionState.latency, norm: sessionState.latency != null ? Math.max(0, 100 - sessionState.latency / 20) : 0, color: '#3ecf8e' },
    ];

    ctx.clearRect(0, 0, W, H);

    const padL = 40, padR = 20, padT = 20, padB = 40;
    const chartW = W - padL - padR;
    const chartH = H - padT - padB;
    const barW = (chartW / bars.length) * 0.55;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + chartH - (i / 4) * chartH;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();

        // Y labels
        ctx.fillStyle = 'rgba(136,146,164,0.8)';
        ctx.font = '10px Inter, system-ui';
        ctx.textAlign = 'right';
        ctx.fillText((i * 25) + '%', padL - 4, y + 4);
    }

    bars.forEach((bar, i) => {
        const x = padL + (chartW / bars.length) * i + (chartW / bars.length - barW) / 2;
        const barH = (bar.norm / 100) * chartH;
        const y = padT + chartH - barH;

        // Bar gradient
        const grad = ctx.createLinearGradient(x, y + barH, x, y);
        grad.addColorStop(0, bar.color + '33');
        grad.addColorStop(1, bar.color + 'cc');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
        ctx.fill();

        // Value label
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        if (bar.raw != null) {
            ctx.fillText(typeof bar.raw === 'number' ? bar.raw.toFixed(1) : bar.raw, x + barW / 2, y - 5);
        }

        // X label
        ctx.fillStyle = 'rgba(136,146,164,0.9)';
        ctx.font = '11px Inter, system-ui';
        ctx.fillText(bar.label, x + barW / 2, padT + chartH + 18);
    });
}

// ═══════════════════════════════════════════════════════════════════
// 11. UTILITY HELPERS
// ═══════════════════════════════════════════════════════════════════

/** Converts a base64 string to a Blob. */
function b64toBlob(b64Data, contentType = '') {
    const byteChars = atob(b64Data);
    const byteArrays = [];
    for (let offset = 0; offset < byteChars.length; offset += 512) {
        const slice = byteChars.slice(offset, offset + 512);
        const ints = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) ints[i] = slice.charCodeAt(i);
        byteArrays.push(new Uint8Array(ints));
    }
    return new Blob(byteArrays, { type: contentType });
}

/** Sets a status badge to a given state. */
function setStatus(el, state, text) {
    el.className = `status-badge ${state}`;
    el.textContent = text;
}

/** Briefly shows an error on a status badge then resets. */
function flashError(el, msg) {
    setStatus(el, 'error', msg);
    setTimeout(() => setStatus(el, 'idle', '● Ready'), 4000);
}

/** Shows a loading spinner inside a button. */
function setBtnLoading(btn, isLoading) {
    if (isLoading) {
        // Only capture original label the first time so back-to-back
        // calls don't overwrite it with the spinner HTML.
        if (!btn.dataset.origText) {
            btn.dataset.origText = btn.innerHTML;
        }
        btn.innerHTML = '<span class="spinner"></span> Please wait…';
        btn.disabled = true;
    } else {
        btn.innerHTML = btn.dataset.origText || btn.innerHTML;
        delete btn.dataset.origText;
        btn.disabled = false;
    }
}

/** Parses a fetch response, throwing a clear error on failure. */
async function handleResponse(res) {
    if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const j = await res.json(); msg = j.error || msg; } catch { /* ignore */ }
        throw new Error(msg);
    }
    return res.json();
}

/** HTML-escapes a string to prevent XSS. */
function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Sets up a clickable + drag-and-drop file input zone.
 * @param {HTMLElement} zone       - The visible drop area
 * @param {HTMLInputElement} input - The hidden file input
 * @param {HTMLElement} label      - Element to show filename in
 * @param {string[]} allowedTypes  - Accepted MIME prefixes or extensions
 */
function setupDropZone(zone, input, label, allowedTypes) {
    // Click to open file picker
    zone.addEventListener('click', () => input.click());

    // Keyboard a11y
    zone.setAttribute('tabindex', '0');
    zone.setAttribute('role', 'button');
    zone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') input.click(); });

    // Drag events
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && isAllowed(file, allowedTypes)) {
            setFileInput(input, file, label);
        } else {
            alert(`Unsupported file type. Accepted: ${allowedTypes.join(', ')}`);
        }
    });

    // Native file input change
    input.addEventListener('change', () => {
        if (input.files[0]) showFilename(input.files[0], label, input);
    });
}

function isAllowed(file, types) {
    return types.some(t => file.type.startsWith(t) || file.name.endsWith(t));
}

function setFileInput(input, file, label) {
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
}

function showFilename(file, labelEl, inputEl) {
    labelEl.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span>📎 ${escHtml(file.name)} <span style="opacity:0.6; font-size:0.85em;">(${(file.size / 1024).toFixed(1)} KB)</span></span>
        </div>
    `;
    const clearBtn = document.createElement('button');
    clearBtn.innerHTML = '✖';
    clearBtn.className = 'btn btn-secondary';
    clearBtn.style.cssText = 'padding: 0.1rem 0.4rem; font-size: 0.7rem; margin-left: 0.8rem; border-radius: 4px;';
    clearBtn.title = 'Clear file';
    clearBtn.onclick = (e) => {
        e.stopPropagation();
        if (inputEl) inputEl.value = '';
        labelEl.classList.add('hidden');
        labelEl.innerHTML = '';
        
        if (inputEl && inputEl.id === 'media-input') {
            const preview = document.getElementById('original-preview');
            const placeholder = document.getElementById('original-placeholder');
            if (preview && placeholder) {
                preview.classList.add('hidden');
                placeholder.classList.remove('hidden');
                preview.src = '';
            }
        }
    };
    labelEl.firstElementChild.appendChild(clearBtn);
    labelEl.classList.remove('hidden');
}

// ═══════════════════════════════════════════════════════════════════
// INIT on DOMContentLoaded
// ═══════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    checkApiHealth();
    // Re-check health every 30 seconds
    setInterval(checkApiHealth, 30_000);
    // Draw empty chart immediately
    renderMetricsChart();
});