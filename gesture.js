import { GestureRecognizer, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task";

// Friendly labels for gestures
const GESTURE_LABELS = {
  None: "—",
  Closed_Fist: "✊ Fist",
  Open_Palm: "🖐️ Open Palm",
  Pointing_Up: "☝️ Pointing Up",
  Thumb_Down: "👎 Thumbs Down",
  Thumb_Up: "👍 Thumbs Up",
  Victory: "✌️ Victory",
  ILoveYou: "🤟 I Love You",
};

let gestureRecognizer = null;
let lastVideoTime = -1;
let animationId = null;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");

async function init() {
  try {
    statusEl.textContent = "Loading model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      numHands: 2,
    });
    statusEl.textContent = "Starting camera...";
    await startCamera();
    statusEl.textContent = "Show your hand to see the gesture";
    detect();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
  });
  video.srcObject = stream;
  await video.play();
}

function detect() {
  if (!gestureRecognizer || video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now() / 1000;
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = gestureRecognizer.recognizeForVideo(video, now);

    if (results.gestures?.length > 0 && results.gestures[0].length > 0) {
      const top = results.gestures[0][0];
      const name = top.categoryName;
      const label = GESTURE_LABELS[name] ?? name;
      overlay.textContent = label;
    } else {
      overlay.textContent = "—";
    }
  }

  animationId = requestAnimationFrame(detect);
}

init();
