import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let handLandmarker = null;
let shared = null;
let animationId = null;
let lastVideoTime = -1;
let uiSetup = false;

let currentEffect = "sine";

// Control state for each hand — values are lerped each frame for stability
const controls = {
  left:  { active: false, pinch: 0, x: 0.5, y: 0.5 },
  right: { active: false, pinch: 0, x: 0.5, y: 0.5 },
};

const LERP = 0.18;

function lerp(a, b, t) { return a + (b - a) * t; }

function dist2D(ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  return Math.sqrt(dx * dx + dy * dy);
}

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

  shared.statusEl.textContent = "Show both hands — pinch controls the effect";
  lastVideoTime = -1;
  detect();
}

export function deactivate() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  const v = shared?.video;
  if (v) v.removeEventListener("videocanvasresize", onVideoCanvasResize);
  shared = null;
}

function setupUI() {
  document.querySelectorAll("#pinch-modes .mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentEffect = btn.dataset.mode;
      document.querySelectorAll("#pinch-modes .mode-btn").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      document.querySelectorAll(".pinch-submenu").forEach((el) =>
        el.classList.add("hidden")
      );
      document.getElementById(`pinch-sub-${currentEffect}`)?.classList.remove("hidden");
    });
  });
}

// ─── Control extraction ───────────────────────────────────────────────────────

function extractControls(results) {
  controls.left.active = false;
  controls.right.active = false;

  if (!results.landmarks?.length) return;

  results.landmarks.forEach((lm, i) => {
    const label = results.handedness[i]?.[0]?.categoryName ?? "";
    // MediaPipe "Left" is camera-left = user's right visually (mirrored feed).
    // Map so controls.left = hand on the visual left of the screen.
    const key = label === "Right" ? "left" : "right";
    if (controls[key].active) return; // already set (shouldn't happen with 2 hands)

    controls[key].active = true;
    const c = controls[key];

    const thumb     = lm[4];
    const index     = lm[8];
    const wrist     = lm[0];
    const middleMCP = lm[9];

    // Normalize pinch by hand scale so it's consistent regardless of camera distance
    const handScale = dist2D(wrist.x, wrist.y, middleMCP.x, middleMCP.y);
    const rawPinch  = dist2D(thumb.x, thumb.y, index.x, index.y);
    const pinch     = Math.min(1, rawPinch / (handScale * 1.4));

    const mx = (thumb.x + index.x) / 2;
    const my = (thumb.y + index.y) / 2;

    c.pinch = lerp(c.pinch, pinch, LERP);
    c.x     = lerp(c.x,     mx,    LERP);
    c.y     = lerp(c.y,     my,    LERP);
  });
}

// ─── Hand overlay ─────────────────────────────────────────────────────────────

function drawHandOverlay(ctx, w, h) {
  ctx.save();
  ctx.textBaseline = "middle";
  ctx.font = "bold 11px monospace";

  for (const [key, c] of Object.entries(controls)) {
    if (!c.active) continue;
    const px = c.x * w;
    const py = c.y * h;
    const color = key === "left" ? "#818cf8" : "#34d399";
    const outerR = 10 + c.pinch * 26;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, outerR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = color + "cc";
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.fillText(
      `${key[0].toUpperCase()} ${(c.pinch * 100).toFixed(0)}%`,
      px + outerR + 5,
      py
    );
  }
  ctx.restore();
}

// ─── Effect: Sine Wave ────────────────────────────────────────────────────────
// Left pinch  → frequency (1–10 cycles)
// Right pinch → amplitude (5–45% of height)
// Left X      → phase shift
// Right X     → hue

function effectSine(ctx, video, w, h) {
  const L = controls.left;
  const R = controls.right;

  ctx.drawImage(video, 0, 0, w, h);

  const freq  = 1 + L.pinch * 9;
  const amp   = h * (0.05 + R.pinch * 0.40);
  const phase = L.x * Math.PI * 6;
  const hue   = Math.round(R.x * 360);

  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = `hsl(${hue},90%,65%)`;
  ctx.shadowColor  = `hsl(${hue},90%,65%)`;
  ctx.shadowBlur   = 14;

  ctx.beginPath();
  for (let px = 0; px <= w; px++) {
    const t  = (px / w) * Math.PI * 2 * freq + phase;
    const py = h / 2 + Math.sin(t) * amp;
    px === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();

  drawHUD(ctx, [
    `freq   ${freq.toFixed(1)} Hz`,
    `amp    ${((amp / h) * 100).toFixed(0)}%`,
    `hue    ${hue}°`,
  ]);
}

// ─── Effect: Lissajous ────────────────────────────────────────────────────────
// Left pinch  → X frequency (1–8)
// Right pinch → Y frequency (1–8)
// Left X      → X phase offset
// Right X     → Y phase offset
// Time drives continuous animation so the curve stays alive

function effectLissajous(ctx, video, w, h) {
  const L = controls.left;
  const R = controls.right;

  ctx.drawImage(video, 0, 0, w, h);
  // Dark overlay so the curve pops
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, 0, w, h);

  const freqX  = 1 + L.pinch * 7;
  const freqY  = 1 + R.pinch * 7;
  const phaseX = L.x * Math.PI * 2;
  const phaseY = R.x * Math.PI * 2;
  const drift  = (performance.now() / 8000) * Math.PI * 2; // slow time drift
  const rx     = w * 0.42;
  const ry     = h * 0.42;

  const STEPS = 2400;

  const grad = ctx.createLinearGradient(-rx, 0, rx, 0);
  grad.addColorStop(0,   "#6366f1");
  grad.addColorStop(0.5, "#34d399");
  grad.addColorStop(1,   "#f472b6");

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.strokeStyle = grad;
  ctx.lineWidth   = 2;
  ctx.shadowColor = "#818cf8";
  ctx.shadowBlur  = 10;

  ctx.beginPath();
  for (let i = 0; i <= STEPS; i++) {
    const t  = (i / STEPS) * Math.PI * 2;
    const px = Math.sin(freqX * t + phaseX + drift) * rx;
    const py = Math.sin(freqY * t + phaseY)          * ry;
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();

  drawHUD(ctx, [
    `x-freq  ${freqX.toFixed(1)}`,
    `y-freq  ${freqY.toFixed(1)}`,
  ]);
}

// ─── Effect: Mirror Split ─────────────────────────────────────────────────────
// Left hand X  → vertical mirror line
// Right hand Y → horizontal mirror line
// Each hand independently activates its axis

function effectMirror(ctx, video, w, h) {
  const L = controls.left;
  const R = controls.right;

  ctx.drawImage(video, 0, 0, w, h);

  if (L.active) {
    const splitX = L.x * w;

    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, w - splitX, h);
    ctx.clip();
    ctx.translate(splitX * 2, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, h);
    ctx.stroke();
    ctx.restore();
  }

  if (R.active) {
    const splitY = R.y * h;

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, splitY, w, h - splitY);
    ctx.clip();
    ctx.translate(0, splitY * 2);
    ctx.scale(1, -1);
    ctx.drawImage(video, 0, 0, w, h);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, splitY);
    ctx.lineTo(w, splitY);
    ctx.stroke();
    ctx.restore();
  }

  if (!L.active && !R.active) {
    drawHUD(ctx, ["Raise a hand to activate"]);
  }
}

// ─── Shared HUD ───────────────────────────────────────────────────────────────

function drawHUD(ctx, lines) {
  const pad = 8;
  const lh  = 16;
  const boxH = pad * 2 + lines.length * lh;
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.52)";
  ctx.fillRect(8, 8, 178, boxH);
  ctx.fillStyle = "#e4e4e7";
  ctx.font = "12px monospace";
  ctx.textBaseline = "top";
  lines.forEach((line, i) => ctx.fillText(line, 14, 14 + i * lh));
  ctx.restore();
}

// ─── Detection loop ───────────────────────────────────────────────────────────

const EFFECTS = {
  sine:      effectSine,
  lissajous: effectLissajous,
  mirror:    effectMirror,
};

function detect() {
  if (!shared) return;
  if (!handLandmarker || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now();
  if (lastVideoTime !== shared.video.currentTime) {
    lastVideoTime = shared.video.currentTime;
    const results = handLandmarker.detectForVideo(shared.video, now);
    extractControls(results);
  }

  const { video, canvas, ctx } = shared;
  const w = canvas.width;
  const h = canvas.height;

  (EFFECTS[currentEffect] ?? effectSine)(ctx, video, w, h);
  drawHandOverlay(ctx, w, h);

  animationId = requestAnimationFrame(detect);
}
