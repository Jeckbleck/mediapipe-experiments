import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

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
    shared.statusEl.textContent = "Loading expression model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  shared.overlay.textContent = "";
  shared.overlay.classList.remove("hidden");
  shared.statusEl.textContent = "Show your face to see expressions";
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

function getExpressionFromBlendshapes(blendshapes) {
  if (!blendshapes?.length) return "Neutral";

  const active = blendshapes
    .filter((b) => b.score >= THRESHOLD)
    .map((b) => ({
      label: BLENDSHAPE_LABELS[b.categoryName] || b.categoryName,
      score: b.score,
    }))
    .sort((a, b) => b.score - a.score);

  if (!active.length) return "Neutral";

  const seen = new Set();
  return (
    active
      .map((a) => a.label)
      .filter((l) => {
        if (seen.has(l)) return false;
        seen.add(l);
        return true;
      })
      .slice(0, 2)
      .join(" + ") || "Neutral"
  );
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

    if (results.faceLandmarks?.length > 0 && results.faceBlendshapes?.length > 0) {
      shared.overlay.textContent = getExpressionFromBlendshapes(
        results.faceBlendshapes[0].categories
      );
    } else {
      shared.overlay.textContent = "No face detected";
    }
  }

  animationId = requestAnimationFrame(detect);
}
