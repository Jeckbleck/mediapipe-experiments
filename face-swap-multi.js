import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { swapFaces } from "./lib/faceSwap.js";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let faceLandmarker = null;
let lastVideoTime = -1;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");

async function init() {
  try {
    statusEl.textContent = "Loading model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      numFaces: 2,
    });
    statusEl.textContent = "Starting camera...";
    await startCamera();
    statusEl.textContent = "Show 2 faces to swap them";
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
  await new Promise((resolve, reject) => {
    const onReady = () => video.play().then(resolve).catch(reject);
    if (video.readyState >= 1) {
      onReady();
    } else {
      video.onloadedmetadata = onReady;
    }
  });
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

async function detect() {
  if (!faceLandmarker || video.readyState < 2) {
    requestAnimationFrame(detect);
    return;
  }

  const now = performance.now() / 1000;
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, now);

    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    if (results.faceLandmarks?.length >= 2) {
      try {
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);
        swapFaces(
          ctx,
          video,
          results.faceLandmarks[0],
          results.faceLandmarks[1],
          canvas.width,
          canvas.height
        );
        ctx.restore();
      } catch (e) {
        console.warn("Face swap error:", e);
      }
    }
  }

  requestAnimationFrame(detect);
}

init();
