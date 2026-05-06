import { ImageSegmenter } from "@mediapipe/tasks-vision";
import { getFileset } from "../lib/vision.js";
import { createPersonCutout } from "../lib/segMask.js";

const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

const BACKGROUNDS = [
  { name: "Beach", url: "/backgrounds/beach.jpg" },
  { name: "Office", url: "/backgrounds/office.jpg" },
  { name: "Space", url: "/backgrounds/space.jpg" },
  { name: "Living Room", url: "/backgrounds/living-room.jpg" },
];

const BACKGROUND_CATEGORY = 0;
const cutoutPerson = createPersonCutout(BACKGROUND_CATEGORY);

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

// Separate canvas for the downscale-upscale blur trick on the background
const blurCanvas = document.createElement("canvas");
const blurCtx = blurCanvas.getContext("2d");

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!segmenter) {
    shared.statusEl.textContent = "Loading segmentation model...";
    const vision = await getFileset();
    segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_PATH,
        delegate: "GPU",
      },
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
  const v = shared?.video;
  if (v) v.removeEventListener("videocanvasresize", onVideoCanvasResize);
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

function applySegmentation(mask) {
  if (!shared) return;
  const { video, canvas, ctx } = shared;
  const w = canvas.width;
  const h = canvas.height;

  if (currentMode === "none") {
    ctx.drawImage(video, 0, 0);
    return;
  }

  const personLayer = cutoutPerson(video, mask, w, h, smoothEdges);

  switch (currentMode) {
    case "blur": {
      // Downscale-upscale trick: GPU bilinear interpolation produces blur without CPU convolution
      const scale = Math.max(4, blurAmount * 1.5);
      const bw = Math.max(2, Math.round(w / scale));
      const bh = Math.max(2, Math.round(h / scale));
      if (blurCanvas.width !== bw || blurCanvas.height !== bh) {
        blurCanvas.width = bw;
        blurCanvas.height = bh;
      }
      blurCtx.drawImage(video, 0, 0, bw, bh);
      ctx.drawImage(blurCanvas, 0, 0, w, h);
      break;
    }
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

  ctx.drawImage(personLayer, 0, 0);
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
