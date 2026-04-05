import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { swapFaces } from "../lib/faceSwap.js";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let faceLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let shared = null;

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!faceLandmarker) {
    shared.statusEl.textContent = "Loading face model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      numFaces: 2,
    });
  }

  shared.statusEl.textContent = "Show 2 faces to swap them";
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
  if (!faceLandmarker || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now();
  if (lastVideoTime !== shared.video.currentTime) {
    lastVideoTime = shared.video.currentTime;
    const results = faceLandmarker.detectForVideo(shared.video, now);

    shared.ctx.drawImage(shared.video, 0, 0);

    if (results.faceLandmarks?.length >= 2) {
      try {
        swapFaces(
          shared.ctx,
          shared.video,
          results.faceLandmarks[0],
          results.faceLandmarks[1],
          shared.canvas.width,
          shared.canvas.height
        );
      } catch (e) {
        console.warn("Face swap error:", e);
      }
    }
  }

  animationId = requestAnimationFrame(detect);
}
