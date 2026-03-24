const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const posePrompt = document.getElementById("pose-prompt");
const photoStripPreview = document.getElementById("photo-strip-preview");
const photoStripCanvases = document.getElementById("photo-strip-canvases");

const featureLoaders = {
  expression: () => import("./features/expression.js"),
  gesture: () => import("./features/gesture.js"),
  "face-mesh": () => import("./features/faceMesh.js"),
  "face-swap-single": () => import("./features/faceSwapSingle.js"),
  "face-swap-multi": () => import("./features/faceSwapMulti.js"),
  segmentation: () => import("./features/segmentation.js"),
};

const moduleCache = {};
let currentModule = null;
let currentFeature = null;
let cameraReady = false;

async function startCamera() {
  if (cameraReady) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
  });
  video.srcObject = stream;
  await new Promise((resolve, reject) => {
    const onReady = () => video.play().then(resolve).catch(reject);
    if (video.readyState >= 1) onReady();
    else video.onloadedmetadata = onReady;
  });
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  cameraReady = true;
}

async function switchFeature(name) {
  if (currentModule) {
    currentModule.deactivate();
    currentModule = null;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  overlay.textContent = "";
  overlay.classList.add("hidden");
  posePrompt.textContent = "";
  posePrompt.classList.add("hidden");
  photoStripPreview.classList.add("hidden");

  document.querySelectorAll(".feature-settings").forEach((el) => el.classList.add("hidden"));
  const settingsEl = document.querySelector(`.feature-settings[data-feature="${name}"]`);
  if (settingsEl) settingsEl.classList.remove("hidden");

  statusEl.textContent = "Loading...";

  try {
    await startCamera();
    if (!moduleCache[name]) {
      moduleCache[name] = await featureLoaders[name]();
    }
    const mod = moduleCache[name];
    currentModule = mod;
    currentFeature = name;

    await mod.activate({
      video,
      canvas,
      ctx,
      overlay,
      statusEl,
      posePrompt,
      photoStripPreview,
      photoStripCanvases,
    });
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

document.querySelectorAll('input[name="feature"]').forEach((radio) => {
  radio.addEventListener("change", (e) => {
    if (e.target.checked) switchFeature(e.target.value);
  });
});
