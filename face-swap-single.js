import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { overlayFace } from "./lib/faceSwap.js";
import { log } from "./lib/logger.js";
import { isBackendAvailable, processFrame } from "./lib/backendApi.js";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Reference face images - Vite serves public/ at root, so use /reference-faces/ for local images
const REFERENCE_IMAGES = [
  { name: "Donnie", url: "/reference-faces/donnie.jpg" },
  { name: "Ye", url: "/reference-faces/ye.jpg" },
  { name: "Person 3", url: "https://randomuser.me/api/portraits/men/67.jpg" },
  { name: "Person 4", url: "https://randomuser.me/api/portraits/women/65.jpg" },
  { name: "Person 5", url: "https://randomuser.me/api/portraits/men/22.jpg" },
  { name: "Person 6", url: "https://randomuser.me/api/portraits/women/89.jpg" },
];

let faceLandmarkerVideo = null;
let faceLandmarkerImage = null;
let lastVideoTime = -1;
let selectedImage = null;
let selectedImageLandmarks = null;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const referenceGrid = document.getElementById("reference-grid");
const debugLogEl = document.getElementById("debug-log");
const troubleshootStatusEl = document.getElementById("troubleshoot-status");

function debugLog(msg) {
  if (debugLogEl) {
    const time = new Date().toLocaleTimeString();
    debugLogEl.textContent = `[${time}] ${msg}\n` + debugLogEl.textContent.slice(0, 500);
  }
}

async function init() {
  try {
    log.info("Initializing face swap single...");
    statusEl.textContent = "Loading model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarkerVideo = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      numFaces: 1,
    });
    faceLandmarkerImage = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "IMAGE",
      numFaces: 1,
    });
    log.info("Models loaded");
    statusEl.textContent = "Starting camera...";
    await startCamera();
    log.info("Camera ready", { canvasWidth: canvas.width, canvasHeight: canvas.height });
    debugLog(`Ready: canvas ${canvas.width}x${canvas.height}`);
    buildReferenceGrid();
    setupTroubleshooting();
    document.getElementById("use-backend")?.addEventListener("change", updateBackendStatus);
    updateBackendStatus().then((ok) => {
      if (ok) document.getElementById("use-backend").checked = true;
    });
    statusEl.textContent = "Select a face, then show yours";
    detect();
  } catch (err) {
    log.error("Init failed", err);
    statusEl.textContent = `Error: ${err.message}`;
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
    log.info("Canvas sized", { w: canvas.width, h: canvas.height });
  } else {
    log.warn("Video has no dimensions", { videoWidth: video.videoWidth, videoHeight: video.videoHeight });
  }
}

function buildReferenceGrid() {
  REFERENCE_IMAGES.forEach((img, i) => {
    const div = document.createElement("div");
    div.className = "reference-option";
    div.dataset.index = i;
    const image = document.createElement("img");
    image.src = img.url;
    image.alt = img.name;
    if (img.url.startsWith("http")) {
      image.crossOrigin = "anonymous";
    }
    div.appendChild(image);
    div.addEventListener("click", () => selectReference(img, image, div));
    referenceGrid.appendChild(div);
  });
}

async function selectReference(img, imageEl, divEl) {
  document.querySelectorAll(".reference-option").forEach((d) => d.classList.remove("selected"));
  divEl.classList.add("selected");

  try {
    log.info("Selecting reference", img.name, img.url);
    statusEl.textContent = "Detecting face in reference...";
    const imgEl = new Image();
    if (img.url.startsWith("http")) {
      imgEl.crossOrigin = "anonymous";
    }
    await new Promise((resolve, reject) => {
      imgEl.onload = () => {
        log.info("Reference image loaded", {
          name: img.name,
          naturalWidth: imgEl.naturalWidth,
          naturalHeight: imgEl.naturalHeight,
        });
        resolve();
      };
      imgEl.onerror = (e) => {
        log.error("Reference image failed to load", img.url, e);
        reject(new Error(`Failed to load ${img.url}`));
      };
      imgEl.src = img.url;
    });

    const results = faceLandmarkerImage.detect(imgEl);
    log.info("Face detection on reference", {
      facesFound: results.faceLandmarks?.length ?? 0,
      landmarksCount: results.faceLandmarks?.[0]?.length,
    });

    if (results.faceLandmarks?.length > 0) {
      selectedImage = imgEl;
      selectedImageLandmarks = results.faceLandmarks[0];
      statusEl.textContent = `Using ${img.name} — show your face`;
      log.info("Reference ready", img.name);
      debugLog(`Reference OK: ${img.name} (${imgEl.naturalWidth}x${imgEl.naturalHeight}), ${results.faceLandmarks[0].length} landmarks`);
    } else {
      statusEl.textContent = "No face found in image. Try another.";
      selectedImage = null;
      selectedImageLandmarks = null;
      log.warn("No face in reference image");
      debugLog("No face found in reference image");
    }
  } catch (e) {
    log.error("Reference image error", e);
    debugLog(`Error: ${e.message}`);
    statusEl.textContent = "Failed to load image. Try another.";
    selectedImage = null;
    selectedImageLandmarks = null;
  }
}

let overlayCallCount = 0;
let lastLogTime = 0;
let lastTroubleshootUpdate = 0;
let lastHasVideoFace = false;
let lastHasReference = false;
let testCanvasUntil = 0;
let lastBackendRequest = 0;
const BACKEND_THROTTLE_MS = 120;

function updateTroubleshootStatus(hasVideoFace, hasReference) {
  if (!troubleshootStatusEl) return;
  const t = performance.now() / 1000;
  if (t - lastTroubleshootUpdate < 0.5) return;
  lastTroubleshootUpdate = t;

  const canvasOk = canvas.width > 0 && canvas.height > 0;
  const videoOk = video.readyState >= 2;

  troubleshootStatusEl.innerHTML = `
    <div>Canvas: ${canvas.width}x${canvas.height} ${canvasOk ? '<span class="ok">✓</span>' : '<span class="fail">✗ zero size</span>'}</div>
    <div>Video ready: ${videoOk ? '<span class="ok">✓</span>' : '<span class="fail">✗</span>'}</div>
    <div>Reference selected: ${hasReference ? '<span class="ok">✓</span>' : '<span class="warn">✗ select one</span>'}</div>
    <div>Your face detected: ${hasVideoFace ? '<span class="ok">✓</span>' : '<span class="warn">✗ show face</span>'}</div>
    <div>Overlay calls: ${overlayCallCount}</div>
  `;
}

async function updateBackendStatus() {
  const el = document.getElementById("backend-status");
  if (!el) return false;
  const ok = await isBackendAvailable();
  el.textContent = ok ? "✓ Python backend connected" : "Run: npm run backend";
  el.style.color = ok ? "#4ade80" : "";
  return ok;
}

function setupTroubleshooting() {
  document.getElementById("btn-test-canvas")?.addEventListener("click", () => {
    testCanvasUntil = performance.now() / 1000 + 5;
    debugLog("Test: Red box will show for 5 sec. If you see it, canvas works.");
  });

  document.getElementById("btn-test-overlay")?.addEventListener("click", () => {
    if (!selectedImage || !selectedImageLandmarks) {
      debugLog("Test: Select a reference image first.");
      return;
    }
    const results = faceLandmarkerVideo.detectForVideo(video, performance.now() / 1000);
    if (!results.faceLandmarks?.length) {
      debugLog("Test: Show your face to the camera first.");
      return;
    }
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0);
    overlayFace(ctx, selectedImage, selectedImageLandmarks, results.faceLandmarks[0], canvas.width, canvas.height);
    ctx.restore();
    debugLog("Test: Overlay drawn once. Check if you see the swapped face.");
  });
}

async function detect() {
  if (!faceLandmarkerVideo || video.readyState < 2) {
    requestAnimationFrame(detect);
    return;
  }

  const useBackend = document.getElementById("use-backend")?.checked && (await isBackendAvailable());
  const hasReference = !!(selectedImage && selectedImageLandmarks);

  if (useBackend && hasReference && Date.now() - lastBackendRequest > BACKEND_THROTTLE_MS) {
    lastBackendRequest = Date.now();
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0);
    ctx.restore();
    const result = await processFrame(canvas, { mode: "face-swap", referenceImage: selectedImage });
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
    const results = faceLandmarkerVideo.detectForVideo(video, now);

    if (canvas.width === 0 || canvas.height === 0) {
      log.warn("Canvas has zero size, skipping draw");
    } else {
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(video, 0, 0);
      if (performance.now() / 1000 < testCanvasUntil) {
        ctx.fillStyle = "rgba(255,0,0,0.9)";
        ctx.fillRect(10, 10, 180, 80);
        ctx.fillStyle = "white";
        ctx.font = "bold 28px sans-serif";
        ctx.fillText("CANVAS OK", 30, 55);
      }
      ctx.restore();
    }

    const hasVideoFace = results.faceLandmarks?.length > 0;
    const hasRef = !!(selectedImage && selectedImageLandmarks);
    lastHasVideoFace = hasVideoFace;
    lastHasReference = hasRef;

    if (!hasVideoFace && hasRef && now - lastLogTime > 3) {
      debugLog("Waiting for your face... (select reference first, then show face)");
      lastLogTime = now;
    }

    if (hasVideoFace && hasRef) {
      overlayCallCount++;
      if (now - lastLogTime > 2) {
        log.info("Overlay active", {
          videoFaces: results.faceLandmarks.length,
          overlayCalls: overlayCallCount,
          canvasSize: `${canvas.width}x${canvas.height}`,
        });
        debugLog(`Overlay: ${results.faceLandmarks.length} face(s), canvas ${canvas.width}x${canvas.height}`);
        lastLogTime = now;
      }
      try {
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);
        overlayFace(
          ctx,
          selectedImage,
          selectedImageLandmarks,
          results.faceLandmarks[0],
          canvas.width,
          canvas.height
        );
        if (window.DEBUG_FACESWAP) {
          const b = results.faceLandmarks[0];
          const minX = Math.min(...b.map((l) => l.x)) * canvas.width;
          const minY = Math.min(...b.map((l) => l.y)) * canvas.height;
          const maxX = Math.max(...b.map((l) => l.x)) * canvas.width;
          const maxY = Math.max(...b.map((l) => l.y)) * canvas.height;
          ctx.fillStyle = "rgba(0,255,0,0.3)";
          ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
        }
        ctx.restore();
      } catch (e) {
        log.error("Face overlay error", e);
        debugLog(`Overlay error: ${e.message}`);
      }
    }
  }

  updateTroubleshootStatus(lastHasVideoFace, lastHasReference);

  requestAnimationFrame(detect);
}

init();
