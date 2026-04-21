import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const STABILITY_DURATION  = 600;   // ms of stillness for both stage 1 and stage 2
const COUNTDOWN_DURATION  = 1000;  // ms
const STABILITY_THRESH    = 0.025; // fraction of canvas diagonal
const MIN_AREA_FRAC       = 0.06;
const STAGE2_DELAY        = 3000;  // ms after freeze before auto-capturing stage 2
const CAPTURED_HOLD       = 2000;  // ms to show composite flash before reset

const STATE = Object.freeze({
  IDLE:      "idle",
  FRAMING:   "framing",
  COUNTDOWN: "countdown",
  FROZEN:    "frozen",    // bg frozen, user moves, waiting for stage 2
  CAPTURED:  "captured",  // composite built, flashing result then reset
});

const STATUS_TEXT = {
  [STATE.IDLE]:      "Frame your shot — extend index fingers and thumbs on both hands",
  [STATE.FRAMING]:   "Hold still…",
  [STATE.COUNTDOWN]: "Freezing background…",
  [STATE.FROZEN]:    "Background frozen — step into the frame",
  [STATE.CAPTURED]:  "Captured!",
};

let handLandmarker = null;
let shared = null;
let animationId = null;
let lastVideoTime = -1;
let uiSetup = false;

let state = STATE.IDLE;
let corners = null;
let stableRef = null;
let stableStart = null;
let countdownStart = null;
// Stage 1
let frozenBg = null;
let frozenCorners = null;
let frozenEnteredAt = null;
// Result
let captureCanvas = null;
let capturedAt    = null;
let prevState     = null;

const captures = [];
let dirHandle = null;

// ─── Lifecycle ────────────────────────────────────────────────────────────────

function onVideoCanvasResize() {
  lastVideoTime = -1;
  if (state === STATE.FROZEN || state === STATE.CAPTURED) resetState();
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!handLandmarker) {
    shared.statusEl.textContent = "Loading hand model…";
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

  resetState();

  const wrap = document.getElementById("frame-carousel-wrap");
  if (wrap) wrap.classList.toggle("hidden", captures.length === 0);

  detect();
}

export function deactivate() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  const v = shared?.video;
  if (v) v.removeEventListener("videocanvasresize", onVideoCanvasResize);
  shared = null;
  resetState();
  document.getElementById("frame-carousel-wrap")?.classList.add("hidden");
}

function setupUI() {
  document.getElementById("frame-reset-btn")?.addEventListener("click", resetState);

  const dirBtn   = document.getElementById("frame-dir-btn");
  const dirLabel = document.getElementById("frame-dir-label");

  if (!("showDirectoryPicker" in window)) {
    document.getElementById("frame-dir-section")?.classList.add("hidden");
  } else {
    dirBtn?.addEventListener("click", async () => {
      try {
        dirHandle = await window.showDirectoryPicker({ mode: "readwrite" });
        if (dirLabel) dirLabel.textContent = dirHandle.name;
      } catch { /* cancelled */ }
    });
  }
}

function resetState() {
  state = STATE.IDLE;
  corners = stableRef = stableStart = countdownStart = null;
  frozenBg = frozenCorners = frozenEnteredAt = null;
  captureCanvas = capturedAt = null;
  prevState = null;
  if (shared) shared.statusEl.textContent = STATUS_TEXT[STATE.IDLE];
}

// ─── Composite builder ────────────────────────────────────────────────────────

function buildComposite(bg, liveFrame, c, w, h) {
  const cc = document.createElement("canvas");
  cc.width  = w;
  cc.height = h;
  const cx  = cc.getContext("2d");

  // Stage 1 — frozen background fills the outside
  cx.drawImage(bg, 0, 0, w, h);

  // Stage 2 — live capture fills the inside, clipped to quad
  cx.save();
  cx.beginPath();
  cx.moveTo(c.tl.x, c.tl.y);
  cx.lineTo(c.tr.x, c.tr.y);
  cx.lineTo(c.br.x, c.br.y);
  cx.lineTo(c.bl.x, c.bl.y);
  cx.closePath();
  cx.clip();
  cx.drawImage(liveFrame, 0, 0, w, h);
  cx.restore();

  // Decorative frame
  drawEdge(cx, c, "rgba(99,102,241,0.85)");
  drawBrackets(cx, c, "#818cf8", 18);

  return cc;
}

// ─── Carousel & save ──────────────────────────────────────────────────────────

async function onCapture(bg, c, videoEl, w, h) {
  const liveFrame = document.createElement("canvas");
  liveFrame.width  = w;
  liveFrame.height = h;
  liveFrame.getContext("2d").drawImage(videoEl, 0, 0, w, h);

  captureCanvas = buildComposite(bg, liveFrame, c, w, h);

  const dataUrl   = captureCanvas.toDataURL("image/jpeg", 0.92);
  const timestamp = new Date();
  captures.push({ dataUrl, timestamp });
  addToCarousel(dataUrl, timestamp);
  if (dirHandle) await saveToDirectory(dataUrl, timestamp).catch(console.warn);
}

function addToCarousel(dataUrl, timestamp) {
  const wrap     = document.getElementById("frame-carousel-wrap");
  const carousel = document.getElementById("frame-carousel");
  if (!wrap || !carousel) return;

  wrap.classList.remove("hidden");

  const item  = document.createElement("div");
  item.className = "frame-carousel-item";
  item.title = "Click to download";

  const img = document.createElement("img");
  img.src = dataUrl;
  img.alt = `Capture ${timestamp.toLocaleTimeString()}`;

  const label = document.createElement("div");
  label.className = "frame-carousel-time";
  label.textContent = timestamp.toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });

  item.appendChild(img);
  item.appendChild(label);
  item.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `frame-${timestamp.getTime()}.jpg`;
    a.click();
  });

  carousel.appendChild(item);
  item.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "end" });
}

async function saveToDirectory(dataUrl, timestamp) {
  const blob     = await (await fetch(dataUrl)).blob();
  const filename = `frame-${timestamp.toISOString().replace(/[:.]/g, "-")}.jpg`;
  const fh       = await dirHandle.getFileHandle(filename, { create: true });
  const writable = await fh.createWritable();
  await writable.write(blob);
  await writable.close();
}

// ─── Geometry ─────────────────────────────────────────────────────────────────

function dist(ax, ay, bx, by) {
  return Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2);
}

function sortCorners(pts) {
  const s  = [...pts].sort((a, b) => a.y - b.y);
  const tl = s[0].x <= s[1].x ? s[0] : s[1];
  const tr = s[0].x <= s[1].x ? s[1] : s[0];
  const bl = s[2].x <= s[3].x ? s[2] : s[3];
  const br = s[2].x <= s[3].x ? s[3] : s[2];
  return { tl, tr, bl, br };
}

function quadArea({ tl, tr, bl, br }) {
  return Math.abs(
    (tl.x * (tr.y - bl.y) +
     tr.x * (br.y - tl.y) +
     br.x * (bl.y - tr.y) +
     bl.x * (tl.y - br.y)) / 2
  );
}

function maxCornerDelta(a, b) {
  return Math.max(
    dist(a.tl.x, a.tl.y, b.tl.x, b.tl.y),
    dist(a.tr.x, a.tr.y, b.tr.x, b.tr.y),
    dist(a.bl.x, a.bl.y, b.bl.x, b.bl.y),
    dist(a.br.x, a.br.y, b.br.x, b.br.y)
  );
}

function cloneCorners(c) {
  return { tl: { ...c.tl }, tr: { ...c.tr }, bl: { ...c.bl }, br: { ...c.br } };
}

function centroid(c) {
  return {
    x: (c.tl.x + c.tr.x + c.br.x + c.bl.x) / 4,
    y: (c.tl.y + c.tr.y + c.br.y + c.bl.y) / 4,
  };
}

// ─── Extraction ───────────────────────────────────────────────────────────────

function extractCorners(results, w, h) {
  if (!results.landmarks || results.landmarks.length < 2) return null;
  const pts = [];
  for (const lm of results.landmarks) {
    pts.push({ x: lm[4].x * w, y: lm[4].y * h });
    pts.push({ x: lm[8].x * w, y: lm[8].y * h });
    if (pts.length === 4) break;
  }
  return pts.length === 4 ? sortCorners(pts) : null;
}

// ─── Drawing ──────────────────────────────────────────────────────────────────

function drawBrackets(ctx, c, color, size = 16) {
  const sides = [
    { p: c.tl, sx:  1, sy:  1 },
    { p: c.tr, sx: -1, sy:  1 },
    { p: c.br, sx: -1, sy: -1 },
    { p: c.bl, sx:  1, sy: -1 },
  ];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = "square";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  for (const { p, sx, sy } of sides) {
    ctx.beginPath();
    ctx.moveTo(p.x + sx * size, p.y);
    ctx.lineTo(p.x, p.y);
    ctx.lineTo(p.x, p.y + sy * size);
    ctx.stroke();
  }
  ctx.restore();
}

function drawEdge(ctx, c, color, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash(dash);
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(c.tl.x, c.tl.y);
  ctx.lineTo(c.tr.x, c.tr.y);
  ctx.lineTo(c.br.x, c.br.y);
  ctx.lineTo(c.bl.x, c.bl.y);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function clipToQuad(ctx, c) {
  ctx.beginPath();
  ctx.moveTo(c.tl.x, c.tl.y);
  ctx.lineTo(c.tr.x, c.tr.y);
  ctx.lineTo(c.br.x, c.br.y);
  ctx.lineTo(c.bl.x, c.bl.y);
  ctx.closePath();
  ctx.clip();
}

function drawStabilityRing(ctx, c, progress, color) {
  const { x, y } = centroid(c);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.arc(x, y, 22, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawCountdownSweep(ctx, c, remaining) {
  const { x, y } = centroid(c);
  const t     = remaining / COUNTDOWN_DURATION;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 70);
  ctx.save();
  ctx.strokeStyle = `rgba(99,102,241,${0.75 + pulse * 0.25})`;
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.shadowColor = "#6366f1";
  ctx.shadowBlur = 18;
  ctx.beginPath();
  ctx.arc(x, y, 28, -Math.PI / 2, -Math.PI / 2 + t * Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawLiveTag(ctx, c) {
  const p     = c.tl;
  const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 500);
  ctx.save();
  ctx.fillStyle = "#ef4444";
  ctx.shadowColor = "#ef4444";
  ctx.shadowBlur = 6 + pulse * 4;
  ctx.beginPath();
  ctx.arc(p.x + 12, p.y + 13, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#e4e4e7";
  ctx.shadowBlur = 0;
  ctx.font = "bold 11px monospace";
  ctx.textBaseline = "middle";
  ctx.fillText("LIVE", p.x + 22, p.y + 13);
  ctx.restore();
}

// ─── Render ───────────────────────────────────────────────────────────────────

function render(video, canvas, ctx) {
  const w   = canvas.width;
  const h   = canvas.height;
  const now = performance.now();
  const diagThreshPx = Math.hypot(w, h) * STABILITY_THRESH;
  const minArea      = w * h * MIN_AREA_FRAC;

  if (state !== prevState) {
    if (shared) shared.statusEl.textContent = STATUS_TEXT[state];
    prevState = state;
  }

  // ── IDLE / FRAMING ──────────────────────────────────────────────────────────
  if (state === STATE.IDLE || state === STATE.FRAMING) {
    ctx.drawImage(video, 0, 0, w, h);

    if (!corners || quadArea(corners) < minArea) {
      if (corners) drawBrackets(ctx, corners, "rgba(255,255,255,0.2)", 12);
      stableRef = null;
      stableStart = null;
      state = STATE.IDLE;
      return;
    }

    if (!stableRef || maxCornerDelta(corners, stableRef) > diagThreshPx) {
      stableRef = cloneCorners(corners);
      stableStart = now;
      state = STATE.FRAMING;
    }

    const progress = Math.min(1, (now - stableStart) / STABILITY_DURATION);
    const hue = Math.round(progress * 120);

    if (progress >= 1) {
      state = STATE.COUNTDOWN;
      countdownStart = now;
      return;
    }

    drawEdge(ctx, corners, `hsla(${hue},85%,60%,0.7)`, [8, 5]);
    drawBrackets(ctx, corners, `hsl(${hue},90%,65%)`);
    drawStabilityRing(ctx, corners, progress, `hsl(${hue},90%,65%)`);

  // ── COUNTDOWN ───────────────────────────────────────────────────────────────
  } else if (state === STATE.COUNTDOWN) {
    ctx.drawImage(video, 0, 0, w, h);

    if (!corners || quadArea(corners) < minArea) {
      state = STATE.IDLE;
      stableRef = null;
      stableStart = null;
      return;
    }

    const remaining = COUNTDOWN_DURATION - (now - countdownStart);

    if (remaining <= 0) {
      // Stage 1: freeze the background
      frozenBg = document.createElement("canvas");
      frozenBg.width  = w;
      frozenBg.height = h;
      frozenBg.getContext("2d").drawImage(video, 0, 0, w, h);
      frozenCorners   = cloneCorners(corners);
      frozenEnteredAt = now;
      state = STATE.FROZEN;
      return;
    }

    const pulse = 0.5 + 0.5 * Math.sin(now / 70);
    ctx.save();
    ctx.fillStyle = `rgba(0,0,0,${0.2 + pulse * 0.1})`;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();

    drawEdge(ctx, corners, `rgba(99,102,241,${0.65 + pulse * 0.35})`);
    drawBrackets(ctx, corners, "#818cf8", 20);
    drawCountdownSweep(ctx, corners, remaining);

  // ── FROZEN ──────────────────────────────────────────────────────────────────
  } else if (state === STATE.FROZEN) {
    ctx.drawImage(frozenBg, 0, 0, w, h);

    // Live window — always clipped to the original frozenCorners, no hand constraint
    ctx.save();
    clipToQuad(ctx, frozenCorners);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    // Glowing frame border
    const glow = 0.5 + 0.5 * Math.sin(now / 700);
    ctx.save();
    ctx.shadowColor = "#6366f1";
    ctx.shadowBlur  = 12 + glow * 8;
    drawEdge(ctx, frozenCorners, `rgba(99,102,241,${0.7 + glow * 0.3})`);
    drawBrackets(ctx, frozenCorners, `rgba(129,140,248,${0.75 + glow * 0.25})`, 16);
    ctx.restore();

    drawLiveTag(ctx, frozenCorners);

    // Stage 2: auto-capture after STAGE2_DELAY, show emerald countdown ring
    const elapsed2 = now - frozenEnteredAt;
    const p2 = Math.min(1, elapsed2 / STAGE2_DELAY);
    drawStabilityRing(ctx, frozenCorners, p2, "#34d399");

    if (p2 >= 1) {
      capturedAt = now;
      state = STATE.CAPTURED;
      onCapture(frozenBg, frozenCorners, video, w, h).catch(console.warn);
      return;
    }

  // ── CAPTURED ────────────────────────────────────────────────────────────────
  } else if (state === STATE.CAPTURED) {
    if (captureCanvas) ctx.drawImage(captureCanvas, 0, 0, w, h);

    const elapsed = now - capturedAt;

    // Brief white flash at the shutter moment
    if (elapsed < 120) {
      ctx.save();
      ctx.fillStyle = `rgba(255,255,255,${0.65 * (1 - elapsed / 120)})`;
      ctx.fillRect(0, 0, w, h);
      ctx.restore();
    }

    if (elapsed > CAPTURED_HOLD) resetState();
  }
}

// ─── Detection loop ───────────────────────────────────────────────────────────

function detect() {
  if (!shared) return;
  if (!handLandmarker || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const { video, canvas, ctx } = shared;
  const now = performance.now();

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = handLandmarker.detectForVideo(video, now);
    corners = extractCorners(results, canvas.width, canvas.height);
  }

  render(video, canvas, ctx);
  animationId = requestAnimationFrame(detect);
}
