import { ImageSegmenter, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

const BACKGROUNDS = [
  { name: "Beach", url: "/backgrounds/beach.jpg" },
  { name: "Office", url: "/backgrounds/office.jpg" },
  { name: "Space", url: "/backgrounds/space.jpg" },
  { name: "Living Room", url: "/backgrounds/living-room.jpg" },
];

const BACKGROUND_CATEGORY = 0;

let segmenter = null;
let shared = null;
let animationId = null;
let lastVideoTime = -1;
let uiSetup = false;

let currentMode = "blur";
let blurAmount = 15;
let bgColor = "#000000";
let bgImage = null;
let smoothEdges = true;

const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
const personCanvas = document.createElement("canvas");
const personCtx = personCanvas.getContext("2d");
let maskImageData = null;

export async function activate(s) {
  shared = s;

  personCanvas.width = shared.canvas.width;
  personCanvas.height = shared.canvas.height;

  if (!segmenter) {
    shared.statusEl.textContent = "Loading segmentation model...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
  }

  if (!uiSetup) {
    setupUI();
    uiSetup = true;
  }

  shared.statusEl.textContent = "Segmentation active";
  lastVideoTime = -1;
  detect();
}

export function deactivate() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  shared = null;
}

function setupUI() {
  document.querySelectorAll("#seg-modes .mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentMode = btn.dataset.mode;
      document.querySelectorAll("#seg-modes .mode-btn").forEach((b) =>
        b.classList.toggle("active", b === btn)
      );
      ["seg-blur", "seg-remove", "seg-image"].forEach((id) => {
        document.getElementById(`submenu-${id}`)?.classList.add("hidden");
      });
      const sub = document.getElementById(`submenu-seg-${currentMode}`);
      if (sub) sub.classList.remove("hidden");
    });
  });

  const blurSlider = document.getElementById("blur-amount");
  const blurValue = document.getElementById("blur-value");
  blurSlider?.addEventListener("input", () => {
    blurAmount = parseInt(blurSlider.value, 10);
    blurValue.textContent = `${blurAmount}px`;
  });

  document.querySelectorAll(".color-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".color-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      bgColor = btn.dataset.color;
    });
  });

  const bgGrid = document.getElementById("bg-grid");
  if (bgGrid) {
    BACKGROUNDS.forEach((bg) => {
      const div = document.createElement("div");
      div.className = "bg-option";
      const img = document.createElement("img");
      img.src = bg.url;
      img.alt = bg.name;
      div.appendChild(img);
      div.addEventListener("click", () => {
        document.querySelectorAll(".bg-option").forEach((d) => d.classList.remove("selected"));
        div.classList.add("selected");
        loadBgImage(bg.url);
      });
      bgGrid.appendChild(div);
    });
  }

  document.getElementById("bg-upload")?.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    loadBgImage(URL.createObjectURL(file));
    document.querySelectorAll(".bg-option").forEach((d) => d.classList.remove("selected"));
  });

  document.getElementById("smooth-edges")?.addEventListener("change", (e) => {
    smoothEdges = e.target.checked;
  });
}

function loadBgImage(url) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => { bgImage = img; };
  img.src = url;
}

function computeLetterbox(maskW, maskH, videoW, videoH) {
  const scale = Math.min(maskW / videoW, maskH / videoH);
  const scaledW = Math.round(videoW * scale);
  const scaledH = Math.round(videoH * scale);
  return {
    offsetX: Math.round((maskW - scaledW) / 2),
    offsetY: Math.round((maskH - scaledH) / 2),
    scaledW,
    scaledH,
  };
}

function updateMaskCanvas(mask) {
  const mw = mask.width;
  const mh = mask.height;

  if (maskCanvas.width !== mw || maskCanvas.height !== mh) {
    maskCanvas.width = mw;
    maskCanvas.height = mh;
    maskImageData = maskCtx.createImageData(mw, mh);
  }

  const maskData = mask.getAsUint8Array();
  const data = maskImageData.data;
  const len = mw * mh;

  for (let i = 0; i < len; i++) {
    const j = i * 4;
    const isPerson = maskData[i] !== BACKGROUND_CATEGORY;
    data[j] = 255;
    data[j + 1] = 255;
    data[j + 2] = 255;
    data[j + 3] = isPerson ? 255 : 0;
  }
  maskCtx.putImageData(maskImageData, 0, 0);
}

function applySegmentation(mask) {
  if (!shared) return;
  const { video, canvas, ctx } = shared;
  const w = canvas.width;
  const h = canvas.height;

  if (currentMode === "none") {
    ctx.drawImage(video, 0, 0);
    return;
  }

  updateMaskCanvas(mask);

  const { offsetX, offsetY, scaledW, scaledH } =
    computeLetterbox(maskCanvas.width, maskCanvas.height, w, h);

  personCtx.save();
  personCtx.clearRect(0, 0, w, h);
  personCtx.drawImage(video, 0, 0);
  personCtx.globalCompositeOperation = "destination-in";
  if (smoothEdges) personCtx.filter = "blur(4px)";
  personCtx.drawImage(maskCanvas, offsetX, offsetY, scaledW, scaledH, 0, 0, w, h);
  personCtx.filter = "none";
  personCtx.globalCompositeOperation = "source-over";
  personCtx.restore();

  switch (currentMode) {
    case "blur":
      ctx.save();
      ctx.filter = `blur(${blurAmount}px)`;
      ctx.drawImage(video, 0, 0);
      ctx.restore();
      break;
    case "remove":
      ctx.fillStyle = bgColor;
      ctx.fillRect(0, 0, w, h);
      break;
    case "image":
      if (bgImage) {
        ctx.drawImage(bgImage, 0, 0, w, h);
      } else {
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, w, h);
      }
      break;
    case "grayscale":
      ctx.save();
      ctx.filter = "grayscale(1)";
      ctx.drawImage(video, 0, 0);
      ctx.restore();
      break;
  }

  ctx.drawImage(personCanvas, 0, 0);
}

function detect() {
  if (!shared) return;
  if (!segmenter || shared.video.readyState < 2) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const now = performance.now();
  if (lastVideoTime !== shared.video.currentTime) {
    lastVideoTime = shared.video.currentTime;
    const result = segmenter.segmentForVideo(shared.video, now);
    if (result.categoryMask) {
      applySegmentation(result.categoryMask);
      result.categoryMask.close();
    } else {
      shared.ctx.drawImage(shared.video, 0, 0);
    }
  }

  animationId = requestAnimationFrame(detect);
}
