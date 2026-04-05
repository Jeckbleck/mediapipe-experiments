import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

/**
 * Each emotion is a weighted sum of blendshape scores.
 * Positive weights are excitatory (signal present → score rises).
 * Negative weights are inhibitory (conflicting signal present → score drops).
 * Scores are clamped to ≥ 0 after summation.
 *
 * Inhibitors solve co-activation problems, e.g. a big smile slightly raises
 * cheeks which nudges browDown, which would otherwise bleed into "angry".
 * With mouthSmile × -0.5 on angry, any real smile drives angry to zero.
 */
const EMOTIONS = {
  happy: {
    emoji: "😊",
    color: "#FFD700",
    weights: {
      mouthSmileLeft:   0.30,
      mouthSmileRight:  0.30,
      cheekSquintLeft:  0.20,
      cheekSquintRight: 0.20,
      // inhibitors
      mouthFrownLeft:  -0.30,
      mouthFrownRight: -0.30,
    },
  },
  sad: {
    emoji: "😢",
    color: "#4A90E2",
    weights: {
      mouthFrownLeft:       0.30,
      mouthFrownRight:      0.30,
      browInnerUp:          0.25,
      mouthLowerDownLeft:   0.075,
      mouthLowerDownRight:  0.075,
      // inhibitors — smile and open jaw both suppress sad
      mouthSmileLeft:  -0.45,
      mouthSmileRight: -0.45,
      jawOpen:         -0.20,
    },
  },
  angry: {
    emoji: "😠",
    color: "#E74C3C",
    weights: {
      browDownLeft:  0.35,
      browDownRight: 0.35,
      noseSneerLeft:  0.15,
      noseSneerRight: 0.15,
      // inhibitors — smile strongly suppresses angry to prevent co-activation
      // when cheek squint from smiling nudges browDown slightly
      mouthSmileLeft:  -0.50,
      mouthSmileRight: -0.50,
      eyeWideLeft:  -0.20,
      eyeWideRight: -0.20,
    },
  },
  surprised: {
    emoji: "😮",
    color: "#F39C12",
    weights: {
      jawOpen:          0.45,
      eyeWideLeft:      0.20,
      eyeWideRight:     0.20,
      browOuterUpLeft:  0.075,
      browOuterUpRight: 0.075,
      // inhibitors — furrowed brows mean anger not surprise
      browDownLeft:  -0.30,
      browDownRight: -0.30,
    },
  },
  disgusted: {
    emoji: "🤢",
    color: "#27AE60",
    weights: {
      noseSneerLeft:  0.35,
      noseSneerRight: 0.35,
      mouthPucker:    0.20,
      browDownLeft:   0.05,
      browDownRight:  0.05,
      // inhibitors — smile and mouth-open both conflict with disgust
      mouthSmileLeft:  -0.60,
      mouthSmileRight: -0.60,
      jawOpen:         -0.30,
    },
  },
  fearful: {
    emoji: "😨",
    color: "#9B59B6",
    weights: {
      eyeWideLeft:      0.20,
      eyeWideRight:     0.20,
      browInnerUp:      0.25,
      browOuterUpLeft:  0.15,
      browOuterUpRight: 0.15,
      mouthStretchLeft:  0.025,
      mouthStretchRight: 0.025,
      // inhibitors — open jaw tips into surprised; smile tips into happy
      jawOpen:         -0.40,
      mouthSmileLeft:  -0.25,
      mouthSmileRight: -0.25,
    },
  },
};

// Score must exceed this to replace "neutral"
const THRESHOLD = 0.15;
// Exponential moving average factor — lower = smoother but slower to react
const SMOOTHING = 0.15;
// Bar chart scale — raw score at which the bar reaches full width
const BAR_SCALE = 0.45;

let faceLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let shared = null;

const smoothed = Object.fromEntries(Object.keys(EMOTIONS).map((k) => [k, 0]));

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!faceLandmarker) {
    shared.statusEl.textContent = "Loading emotion model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  shared.overlay.classList.remove("hidden");
  shared.overlay.textContent = "😐 Neutral";
  shared.statusEl.textContent = "Show your face to detect emotions";
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

function computeRawScores(categories) {
  const bs = {};
  for (const { categoryName, score } of categories) {
    bs[categoryName] = score;
  }
  const raw = {};
  for (const [name, { weights }] of Object.entries(EMOTIONS)) {
    let s = 0;
    for (const [key, w] of Object.entries(weights)) {
      s += (bs[key] ?? 0) * w;
    }
    raw[name] = Math.max(0, s);
  }
  return raw;
}

function updateSmoothed(raw) {
  for (const name of Object.keys(EMOTIONS)) {
    smoothed[name] = smoothed[name] * (1 - SMOOTHING) + raw[name] * SMOOTHING;
  }
}

function getPrimaryEmotion() {
  let best = "neutral";
  let bestScore = THRESHOLD;
  for (const [name, score] of Object.entries(smoothed)) {
    if (score > bestScore) {
      best = name;
      bestScore = score;
    }
  }
  return best;
}

function drawBars(ctx, w, h) {
  const entries = Object.entries(EMOTIONS);
  const barH = 14;
  const gap = 5;
  const labelW = 68;
  const barMaxW = 110;
  const padX = 12;
  const padY = 10;
  const boxW = padX * 2 + labelW + barMaxW;
  const boxH = padY * 2 + entries.length * (barH + gap) - gap;
  // Position from visual left edge. The CSS scaleX(-1) on the canvas element
  // mirrors it for the selfie view, so we apply a counter-transform here so
  // that local coordinates match what the user actually sees.
  const x0 = 10;
  const y0 = h - boxH - 10;

  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);

  // background panel
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  if (ctx.roundRect) {
    ctx.beginPath();
    ctx.roundRect(x0, y0, boxW, boxH, 8);
    ctx.fill();
  } else {
    ctx.fillRect(x0, y0, boxW, boxH);
  }

  entries.forEach(([name, { color, emoji }], i) => {
    const score = smoothed[name] ?? 0;
    const barFill = Math.min(1, score / BAR_SCALE);
    const y = y0 + padY + i * (barH + gap);
    const barX = x0 + padX + labelW;

    // emotion label
    ctx.font = `12px sans-serif`;
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fillText(`${emoji} ${name}`, x0 + padX, y + barH - 2);

    // bar track
    ctx.fillStyle = "rgba(255,255,255,0.15)";
    ctx.fillRect(barX, y, barMaxW, barH);

    // bar fill
    if (barFill > 0) {
      ctx.fillStyle = color;
      ctx.fillRect(barX, y, Math.round(barMaxW * barFill), barH);
    }
  });

  ctx.restore();
}

function detect() {
  if (!shared) return;
  if (!faceLandmarker || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const { video, canvas, ctx, overlay } = shared;
  const now = performance.now();

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, now);

    ctx.drawImage(video, 0, 0);

    if (results.faceBlendshapes?.length > 0) {
      const raw = computeRawScores(results.faceBlendshapes[0].categories);
      updateSmoothed(raw);

      const emotion = getPrimaryEmotion();
      const emoji = emotion === "neutral" ? "😐" : EMOTIONS[emotion].emoji;
      const label = emotion.charAt(0).toUpperCase() + emotion.slice(1);
      overlay.textContent = `${emoji} ${label}`;

      drawBars(ctx, canvas.width, canvas.height);
    } else {
      overlay.textContent = "No face detected";
    }
  }

  animationId = requestAnimationFrame(detect);
}
