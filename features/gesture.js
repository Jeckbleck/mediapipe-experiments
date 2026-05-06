import { GestureRecognizer } from "@mediapipe/tasks-vision";
import { getFileset } from "../lib/vision.js";
import { createPerfMonitor } from "../lib/detectionTimer.js";

const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

const GESTURE_LABELS = {
  None: "—",
  Closed_Fist: "Fist",
  Open_Palm: "Open Palm",
  Pointing_Up: "Pointing Up",
  Thumb_Down: "Thumbs Down",
  Thumb_Up: "Thumbs Up",
  Victory: "Victory",
  ILoveYou: "I Love You",
};

const perf = createPerfMonitor();

let gestureRecognizer = null;
let animationId = null;
let lastVideoTime = -1;
let shared = null;
let lastDetectionMs = 0;

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!gestureRecognizer) {
    shared.statusEl.textContent = "Loading gesture model...";
    const vision = await getFileset();
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }

  shared.overlay.textContent = "—";
  shared.overlay.classList.remove("hidden");
  shared.statusEl.textContent = "Show your hand to see gestures";
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

function detect() {
  if (!shared) return;
  if (!gestureRecognizer || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const { video, canvas, ctx, overlay } = shared;
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(video, 0, 0, w, h);

  const now = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const t0 = performance.now();
    const results = gestureRecognizer.recognizeForVideo(video, now);
    lastDetectionMs = performance.now() - t0;

    if (results.gestures?.length > 0 && results.gestures[0].length > 0) {
      const name = results.gestures[0][0].categoryName;
      overlay.textContent = GESTURE_LABELS[name] ?? name;
    } else {
      overlay.textContent = "—";
    }
  }

  perf.draw(ctx, lastDetectionMs, w, h);
  animationId = requestAnimationFrame(detect);
}
