import { GestureRecognizer, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
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

let gestureRecognizer = null;
let animationId = null;
let lastVideoTime = -1;
let shared = null;

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!gestureRecognizer) {
    shared.statusEl.textContent = "Loading gesture model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
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

  const now = performance.now();
  if (lastVideoTime !== shared.video.currentTime) {
    lastVideoTime = shared.video.currentTime;
    const results = gestureRecognizer.recognizeForVideo(shared.video, now);

    if (results.gestures?.length > 0 && results.gestures[0].length > 0) {
      const name = results.gestures[0][0].categoryName;
      shared.overlay.textContent = GESTURE_LABELS[name] ?? name;
    } else {
      shared.overlay.textContent = "—";
    }
  }

  animationId = requestAnimationFrame(detect);
}
