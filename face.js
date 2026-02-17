import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// Map blendshape names to friendly expression labels
const BLENDSHAPE_LABELS = {
  mouthSmileLeft: "Smile",
  mouthSmileRight: "Smile",
  mouthFrownLeft: "Frown",
  mouthFrownRight: "Frown",
  jawOpen: "Mouth Open",
  mouthPucker: "Kiss",
  mouthFunnel: "Whistle",
  cheekPuff: "Cheek Puff",
  cheekSquintLeft: "Squint",
  cheekSquintRight: "Squint",
  eyeBlinkLeft: "Wink Left",
  eyeBlinkRight: "Wink Right",
  eyeWideLeft: "Wide Eye",
  eyeWideRight: "Surprised",
  browInnerUp: "Surprised",
  browOuterUpLeft: "Eyebrow Up",
  browOuterUpRight: "Eyebrow Up",
  noseSneerLeft: "Nose Scrunch",
  noseSneerRight: "Nose Scrunch",
};

const THRESHOLD = 0.3;

let faceLandmarker = null;
let lastVideoTime = -1;
let animationId = null;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");

async function init() {
  try {
    statusEl.textContent = "Loading model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
    statusEl.textContent = "Starting camera...";
    await startCamera();
    statusEl.textContent = "Show your face to see your expression";
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

function getExpressionFromBlendshapes(blendshapes) {
  if (!blendshapes || blendshapes.length === 0) return "Neutral";

  const active = blendshapes
    .filter((b) => b.score >= THRESHOLD)
    .map((b) => ({
      name: b.categoryName,
      score: b.score,
      label: BLENDSHAPE_LABELS[b.categoryName] || b.categoryName,
    }))
    .sort((a, b) => b.score - a.score);

  if (active.length === 0) return "Neutral";

  // Deduplicate labels (e.g. mouthSmileLeft + mouthSmileRight -> Smile)
  const seen = new Set();
  const labels = active
    .map((a) => a.label)
    .filter((l) => {
      if (seen.has(l)) return false;
      seen.add(l);
      return true;
    });

  return labels.slice(0, 2).join(" + ") || "Neutral";
}

function detect() {
  if (!faceLandmarker || video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now() / 1000;
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, now);

    if (results.faceLandmarks?.length > 0 && results.faceBlendshapes?.length > 0) {
      const blendshapes = results.faceBlendshapes[0].categories;
      const expression = getExpressionFromBlendshapes(blendshapes);
      overlay.textContent = expression;
    } else {
      overlay.textContent = "No face detected";
    }
  }

  animationId = requestAnimationFrame(detect);
}

init();
