import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { swapFaces } from "./lib/faceSwap.js";
import { isBackendAvailable, processFrame } from "./lib/backendApi.js";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let faceLandmarker = null;
let lastVideoTime = -1;
let lastBackendRequest = 0;
const BACKEND_THROTTLE_MS = 120;

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
    document.getElementById("use-backend")?.addEventListener("change", updateBackendStatus);
    updateBackendStatus().then((ok) => {
      if (ok) document.getElementById("use-backend").checked = true;
    });
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

async function updateBackendStatus() {
  const el = document.getElementById("backend-status");
  if (!el) return false;
  const ok = await isBackendAvailable();
  el.textContent = ok ? "✓ Python backend connected" : "Run: npm run backend";
  el.style.color = ok ? "#4ade80" : "";
  return ok;
}

async function detect() {
  if (!faceLandmarker || video.readyState < 2) {
    requestAnimationFrame(detect);
    return;
  }

  const useBackend = document.getElementById("use-backend")?.checked && (await isBackendAvailable());
  if (useBackend && Date.now() - lastBackendRequest > BACKEND_THROTTLE_MS) {
    lastBackendRequest = Date.now();
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0);
    ctx.restore();
    const result = await processFrame(canvas, { mode: "face-swap-multi" });
    if (result.image) {
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = `data:image/jpeg;base64,${result.image}`;
      });
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }
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
