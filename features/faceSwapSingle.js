import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { overlayFace } from "../lib/faceSwap.js";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const REFERENCE_IMAGES = [
  { name: "Donnie", url: "/reference-faces/donnie.jpg" },
  { name: "Ye", url: "/reference-faces/ye.jpg" },
  { name: "Person 3", url: "https://randomuser.me/api/portraits/men/22.jpg" },
  { name: "Person 4", url: "https://randomuser.me/api/portraits/women/89.jpg" },
  { name: "Person 5", url: "https://randomuser.me/api/portraits/men/45.jpg" },
  { name: "Person 6", url: "https://randomuser.me/api/portraits/women/32.jpg" },
];

let faceLandmarkerVideo = null;
let faceLandmarkerImage = null;
let animationId = null;
let lastVideoTime = -1;
let shared = null;

let selectedImage = null;
let selectedImageLandmarks = null;
let gridBuilt = false;

export async function activate(s) {
  shared = s;

  if (!faceLandmarkerVideo) {
    shared.statusEl.textContent = "Loading face swap model...";
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
  }

  if (!gridBuilt) {
    buildReferenceGrid();
    gridBuilt = true;
  }

  shared.statusEl.textContent = selectedImage
    ? "Show your face to swap"
    : "Select a reference face, then show yours";
  lastVideoTime = -1;
  detect();
}

export function deactivate() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  shared = null;
}

function buildReferenceGrid() {
  const grid = document.getElementById("reference-grid");
  if (!grid) return;
  grid.innerHTML = "";

  REFERENCE_IMAGES.forEach((img) => {
    const div = document.createElement("div");
    div.className = "reference-option";
    const image = document.createElement("img");
    image.src = img.url;
    image.alt = img.name;
    image.crossOrigin = "anonymous";
    div.appendChild(image);
    div.addEventListener("click", () => selectReference(img, div));
    grid.appendChild(div);
  });
}

async function selectReference(img, divEl) {
  document.querySelectorAll(".reference-option").forEach((d) => d.classList.remove("selected"));
  divEl.classList.add("selected");

  if (!shared) return;

  try {
    shared.statusEl.textContent = "Detecting face in reference...";
    const imgEl = new Image();
    imgEl.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      imgEl.onload = resolve;
      imgEl.onerror = () => reject(new Error(`Failed to load ${img.url}`));
      imgEl.src = img.url;
    });

    const results = faceLandmarkerImage.detect(imgEl);
    if (results.faceLandmarks?.length > 0) {
      selectedImage = imgEl;
      selectedImageLandmarks = results.faceLandmarks[0];
      shared.statusEl.textContent = `Using ${img.name} — show your face`;
    } else {
      shared.statusEl.textContent = "No face found in image. Try another.";
      selectedImage = null;
      selectedImageLandmarks = null;
    }
  } catch (e) {
    if (shared) shared.statusEl.textContent = "Failed to load image. Try another.";
    selectedImage = null;
    selectedImageLandmarks = null;
  }
}

function detect() {
  if (!shared) return;
  if (!faceLandmarkerVideo || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now();
  if (lastVideoTime !== shared.video.currentTime) {
    lastVideoTime = shared.video.currentTime;
    const results = faceLandmarkerVideo.detectForVideo(shared.video, now);

    shared.ctx.drawImage(shared.video, 0, 0);

    if (results.faceLandmarks?.length > 0 && selectedImage && selectedImageLandmarks) {
      try {
        overlayFace(
          shared.ctx,
          selectedImage,
          selectedImageLandmarks,
          results.faceLandmarks[0],
          shared.canvas.width,
          shared.canvas.height
        );
      } catch (e) {
        console.warn("Face overlay error:", e);
      }
    }
  }

  animationId = requestAnimationFrame(detect);
}
