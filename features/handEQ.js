import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const BANDS = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
const MAX_GAIN_DB = 12;
const LERP = 0.12;

// Left hand (MediaPipe "Right") → bands 0-4: pinky→thumb (low to mid-low)
// Right hand (MediaPipe "Left") → bands 5-9: thumb→pinky (mid-high to high)
const LEFT_TIPS  = [20, 16, 12, 8, 4]; // pinky→thumb tip indices
const RIGHT_TIPS = [4,  8, 12, 16, 20]; // thumb→pinky tip indices

let handLandmarker = null;
let shared = null;
let animationId = null;
let lastVideoTime = -1;
let uiSetup = false;

// Audio state
let audioCtx = null;
let sourceNode = null;
let filters = [];
let mediaStream = null;

// EQ gains (dB), persists across hand loss
const gains = new Array(10).fill(0);
// Smoothed gains for display
const smoothGains = new Array(10).fill(0);

function lerp(a, b, t) { return a + (b - a) * t; }

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!handLandmarker) {
    shared.statusEl.textContent = "Loading hand model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }

  if (!uiSetup) {
    setupUI();
    uiSetup = true;
  }

  shared.statusEl.textContent = "Connect audio source to begin";
  lastVideoTime = -1;
  detect();
}

export function deactivate() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  const v = shared?.video;
  if (v) v.removeEventListener("videocanvasresize", onVideoCanvasResize);
  shared = null;
  disconnectAudio();
}

// ─── Audio setup ──────────────────────────────────────────────────────────────

async function connectAudio() {
  if (audioCtx) return;

  try {
    mediaStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: { echoCancellation: false, noiseSuppression: false, sampleRate: 48000 },
    });

    // Drop video tracks — we only want audio
    mediaStream.getVideoTracks().forEach((t) => t.stop());

    const audioTracks = mediaStream.getAudioTracks();
    if (!audioTracks.length) {
      shared?.statusEl && (shared.statusEl.textContent = "No audio track — share a tab with audio");
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
      return;
    }

    audioCtx = new AudioContext();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);

    filters = BANDS.map((freq, i) => {
      const f = audioCtx.createBiquadFilter();
      f.type = "peaking";
      f.frequency.value = freq;
      f.Q.value = 1.41;
      f.gain.value = gains[i];
      return f;
    });

    // Chain: source → f0 → f1 → … → f9 → destination
    sourceNode.connect(filters[0]);
    for (let i = 0; i < filters.length - 1; i++) filters[i].connect(filters[i + 1]);
    filters[filters.length - 1].connect(audioCtx.destination);

    mediaStream.getAudioTracks()[0].addEventListener("ended", disconnectAudio);

    updateConnectButton(true);
    shared?.statusEl && (shared.statusEl.textContent = "Audio connected — raise hands to EQ");
  } catch (err) {
    shared?.statusEl && (shared.statusEl.textContent = `Audio error: ${err.message}`);
    console.error(err);
  }
}

function disconnectAudio() {
  if (filters.length) {
    try { filters[filters.length - 1].disconnect(); } catch (_) {}
    filters = [];
  }
  if (sourceNode) { try { sourceNode.disconnect(); } catch (_) {} sourceNode = null; }
  if (audioCtx)   { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  updateConnectButton(false);
  shared?.statusEl && (shared.statusEl.textContent = "Audio disconnected");
}

function updateConnectButton(connected) {
  const btn = document.getElementById("eq-connect-btn");
  if (!btn) return;
  btn.textContent = connected ? "Disconnect" : "Connect audio";
  btn.classList.toggle("active", connected);
}

function flattenGains() {
  gains.fill(0);
  smoothGains.fill(0);
  if (filters.length) filters.forEach((f) => (f.gain.value = 0));
}

// ─── UI ───────────────────────────────────────────────────────────────────────

function setupUI() {
  document.getElementById("eq-connect-btn")?.addEventListener("click", () => {
    if (audioCtx) disconnectAudio();
    else connectAudio();
  });
  document.getElementById("eq-flat-btn")?.addEventListener("click", flattenGains);
}

// ─── Hand control extraction ──────────────────────────────────────────────────

function extractGains(results) {
  if (!results.landmarks?.length) return;

  results.landmarks.forEach((lm, i) => {
    const label = results.handedness[i]?.[0]?.categoryName ?? "";
    // MediaPipe "Left"  = user's left hand  → bands 0-4 (low,  pinky→thumb: 31 Hz–500 Hz)
    // MediaPipe "Right" = user's right hand → bands 5-9 (high, thumb→pinky: 1 kHz–16 kHz)
    // Thumbs meet at the 500 Hz / 1 kHz boundary; pinkies hold the outer extremes
    if (label === "Left") {
      LEFT_TIPS.forEach((tipIdx, band) => {
        const raw = (0.5 - lm[tipIdx].y) * 2 * MAX_GAIN_DB;
        gains[band] = Math.max(-MAX_GAIN_DB, Math.min(MAX_GAIN_DB, raw));
      });
    } else if (label === "Right") {
      RIGHT_TIPS.forEach((tipIdx, band) => {
        const raw = (0.5 - lm[tipIdx].y) * 2 * MAX_GAIN_DB;
        gains[5 + band] = Math.max(-MAX_GAIN_DB, Math.min(MAX_GAIN_DB, raw));
      });
    }
  });

  // Push to filter nodes
  if (filters.length) {
    filters.forEach((f, i) => (f.gain.value = gains[i]));
  }
}

// ─── EQ visualizer ───────────────────────────────────────────────────────────

const BAR_COLORS = [
  "#6366f1","#7c73f5","#818cf8","#93c5fd","#67e8f9",
  "#34d399","#6ee7b7","#fbbf24","#fb923c","#f87171",
];

function drawEQPanel(ctx, w, h) {
  const panelH = Math.round(h * 0.28);
  const panelY = h - panelH;
  const padX   = 32;
  const padTop = 16;
  const padBot = 26;
  const innerW = w - padX * 2;
  const innerH = panelH - padTop - padBot;
  const n = BANDS.length;
  const bandW = innerW / n;

  // Panel background (drawn before the counter-transform so it covers the full width)
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.68)";
  ctx.fillRect(0, panelY, w, panelH);

  // Counter the CSS scaleX(-1) so the EQ reads left=low, right=high
  ctx.transform(-1, 0, 0, 1, w, 0);

  // dB grid lines
  const dbLines = [-12, -6, 0, 6, 12];
  ctx.lineWidth = 0.5;
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  dbLines.forEach((db) => {
    const gy = panelY + padTop + innerH * (1 - (db + MAX_GAIN_DB) / (MAX_GAIN_DB * 2));
    ctx.strokeStyle = db === 0 ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.1)";
    ctx.beginPath();
    ctx.moveTo(padX, gy);
    ctx.lineTo(w - padX, gy);
    ctx.stroke();
    ctx.fillStyle = "rgba(200,200,200,0.55)";
    ctx.fillText(`${db > 0 ? "+" : ""}${db}`, padX - 3, gy);
  });

  // Bars + smooth gains + labels
  for (let i = 0; i < n; i++) {
    smoothGains[i] = lerp(smoothGains[i], gains[i], LERP * 3);
    const g  = smoothGains[i];
    const cx = padX + (i + 0.5) * bandW;
    const zeroY = panelY + padTop + innerH * 0.5;
    const barH  = (g / MAX_GAIN_DB) * (innerH * 0.5);
    const barY  = g >= 0 ? zeroY - barH : zeroY;
    const color = BAR_COLORS[i];

    // Bar fill
    ctx.fillStyle = color + "99";
    ctx.fillRect(cx - bandW * 0.28, Math.min(zeroY, barY), bandW * 0.56, Math.abs(barH));

    // Freq label
    const freq = BANDS[i];
    const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
    ctx.fillStyle = "#e4e4e7";
    ctx.font = "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(label, cx, panelY + panelH - padBot + 5);
  }

  // Smooth curve through bar tops
  const points = [];
  for (let i = 0; i < n; i++) {
    const g  = smoothGains[i];
    const cx = padX + (i + 0.5) * bandW;
    const cy = panelY + padTop + innerH * (1 - (g + MAX_GAIN_DB) / (MAX_GAIN_DB * 2));
    points.push([cx, cy]);
  }

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#e4e4e7cc";
  ctx.shadowColor = "#818cf8";
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 0; i < points.length - 1; i++) {
    const mx = (points[i][0] + points[i + 1][0]) / 2;
    ctx.bezierCurveTo(mx, points[i][1], mx, points[i + 1][1], points[i + 1][0], points[i + 1][1]);
  }
  ctx.stroke();

  ctx.restore();
}

// ─── Hand overlay dots ────────────────────────────────────────────────────────

function drawHandDots(ctx, results, w, h) {
  if (!results.landmarks?.length) return;
  results.landmarks.forEach((lm, i) => {
    const label = results.handedness[i]?.[0]?.categoryName ?? "";
    const tips  = label === "Right" ? LEFT_TIPS : RIGHT_TIPS;
    const color = label === "Left" ? "#818cf8" : "#34d399";
    tips.forEach((idx) => {
      ctx.save();
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(lm[idx].x * w, lm[idx].y * h, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  });
}

// ─── Detection loop ───────────────────────────────────────────────────────────

function detect() {
  if (!shared) return;
  if (!handLandmarker || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now();
  let results = null;
  if (lastVideoTime !== shared.video.currentTime) {
    lastVideoTime = shared.video.currentTime;
    results = handLandmarker.detectForVideo(shared.video, now);
    extractGains(results);
  }

  const { video, canvas, ctx } = shared;
  const w = canvas.width;
  const h = canvas.height;

  ctx.drawImage(video, 0, 0, w, h);
  if (results) drawHandDots(ctx, results, w, h);
  drawEQPanel(ctx, w, h);

  if (!audioCtx) {
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, w, h - Math.round(h * 0.28));
    ctx.transform(-1, 0, 0, 1, w, 0);
    ctx.fillStyle = "#e4e4e7";
    ctx.font = "bold 16px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Connect audio to start equalizing", w / 2, (h - Math.round(h * 0.28)) / 2);
    ctx.restore();
  }

  animationId = requestAnimationFrame(detect);
}
