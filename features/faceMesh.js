import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const L = {
  leftEyeOuter: 33,
  leftEyeInner: 133,
  rightEyeOuter: 263,
  rightEyeInner: 362,
  noseTip: 4,
  noseBridge: 6,
  leftMouth: 61,
  rightMouth: 291,
  upperLip: 13,
  forehead: 10,
  chin: 152,
  leftFace: 234,
  rightFace: 454,
};

// ---------------------------------------------------------------------------
// AR prop sprites — inline SVG data URIs, no external assets needed.
// Each SVG has a documented anchor point (the pixel in SVG space that gets
// pinned to the face landmark). drawPropImage() handles translate+rotate.
// ---------------------------------------------------------------------------
const PROP_SVGS = {
  // anchor: (110, 44) — nose-bridge centre
  glasses: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 90">
    <circle cx="57" cy="46" r="38" fill="rgba(180,215,255,0.13)" stroke="#1c1c1c" stroke-width="6.5"/>
    <circle cx="163" cy="46" r="38" fill="rgba(180,215,255,0.13)" stroke="#1c1c1c" stroke-width="6.5"/>
    <path d="M95 42 Q110 34 125 42" fill="none" stroke="#1c1c1c" stroke-width="5" stroke-linecap="round"/>
    <line x1="20" y1="33" x2="0" y2="27" stroke="#1c1c1c" stroke-width="6.5" stroke-linecap="round"/>
    <line x1="200" y1="33" x2="220" y2="27" stroke="#1c1c1c" stroke-width="6.5" stroke-linecap="round"/>
  </svg>`,

  // anchor: (110, 40) — nose-bridge centre
  sunglasses: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 88">
    <path d="M9 18 L97 18 L97 58 Q88 76 50 76 Q11 72 9 54 Z" fill="#0d0d0d" stroke="#000" stroke-width="4" stroke-linejoin="round"/>
    <path d="M211 18 L123 18 L123 58 Q132 76 170 76 Q209 72 211 54 Z" fill="#0d0d0d" stroke="#000" stroke-width="4" stroke-linejoin="round"/>
    <path d="M97 36 Q110 28 123 36" fill="none" stroke="#000" stroke-width="6" stroke-linecap="round"/>
    <line x1="9" y1="29" x2="0" y2="24" stroke="#000" stroke-width="6" stroke-linecap="round"/>
    <line x1="211" y1="29" x2="220" y2="24" stroke="#000" stroke-width="6" stroke-linecap="round"/>
    <path d="M17 24 L87 24 L88 37 Q52 33 15 39 Z" fill="rgba(255,255,255,0.07)"/>
    <path d="M203 24 L133 24 L132 37 Q168 33 205 39 Z" fill="rgba(255,255,255,0.07)"/>
  </svg>`,

  // anchor: (120, 183) — bottom of brim (169+14)
  hat: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 185">
    <rect x="60" y="12" width="120" height="140" rx="5" fill="#141414" stroke="#0a0a0a" stroke-width="3"/>
    <ellipse cx="120" cy="14" rx="61" ry="9" fill="#1e1e1e"/>
    <rect x="60" y="138" width="120" height="20" fill="#7B0000"/>
    <line x1="62" y1="152" x2="178" y2="152" stroke="rgba(255,200,200,0.15)" stroke-width="1.5"/>
    <ellipse cx="120" cy="172" rx="108" ry="14" fill="#0c0c0c"/>
    <ellipse cx="120" cy="169" rx="108" ry="14" fill="#181818" stroke="#0a0a0a" stroke-width="2"/>
    <path d="M76 18 L164 18" stroke="rgba(255,255,255,0.06)" stroke-width="6" stroke-linecap="round"/>
  </svg>`,

  // anchor: (100, 126) — bottom of base band (100+26)
  crown: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 130">
    <path d="M5,122 L5,62 L36,94 L62,16 L84,70 L100,6 L116,70 L138,16 L164,94 L195,62 L195,122 Z"
          fill="#FFD700" stroke="#CC9900" stroke-width="3" stroke-linejoin="round"/>
    <rect x="5" y="100" width="190" height="26" rx="4" fill="#F0C000" stroke="#CC9900" stroke-width="2"/>
    <circle cx="100" cy="114" r="8"  fill="#C0392B" stroke="#922B21" stroke-width="1.5"/>
    <circle cx="55"  cy="112" r="6"  fill="#2471A3" stroke="#1A5276" stroke-width="1.5"/>
    <circle cx="145" cy="112" r="6"  fill="#1E8449" stroke="#186A3B" stroke-width="1.5"/>
    <circle cx="97"  cy="111" r="2.5" fill="rgba(255,255,255,0.7)"/>
    <circle cx="52"  cy="109" r="2"   fill="rgba(255,255,255,0.7)"/>
    <circle cx="142" cy="109" r="2"   fill="rgba(255,255,255,0.7)"/>
  </svg>`,

  // anchor: (90, 10) — top centre (mustache drops down from here)
  mustache: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 72">
    <path d="M90,10 C72,10 50,16 32,32 C18,44 8,58 4,66
             C14,70 26,62 40,46 C52,32 68,20 82,16
             L90,14 L98,16
             C112,20 128,32 140,46 C154,62 166,70 176,66
             C172,58 162,44 148,32 C130,16 108,10 90,10 Z"
          fill="#2c1c0e" stroke="#1a0f06" stroke-width="2"/>
    <path d="M4 66 Q-2 58 3 50 Q9 46 15 53"
          fill="none" stroke="#2c1c0e" stroke-width="5.5" stroke-linecap="round"/>
    <path d="M176 66 Q182 58 177 50 Q171 46 165 53"
          fill="none" stroke="#2c1c0e" stroke-width="5.5" stroke-linecap="round"/>
  </svg>`,

  // anchor: (80, 50) — visual centre of bow
  bow: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 100">
    <path d="M80,50 C60,28 16,12 3,24 C-3,34 8,56 30,62 C52,68 72,58 80,50 Z"
          fill="#E91E8C" stroke="#C2185B" stroke-width="2.5"/>
    <path d="M80,50 C100,28 144,12 157,24 C163,34 152,56 130,62 C108,68 88,58 80,50 Z"
          fill="#E91E8C" stroke="#C2185B" stroke-width="2.5"/>
    <ellipse cx="80" cy="50" rx="13" ry="13" fill="#C2185B" stroke="#AD1457" stroke-width="2"/>
    <path d="M80,50 C64,32 26,18 8,27"
          fill="none" stroke="rgba(255,255,255,0.22)" stroke-width="7" stroke-linecap="round"/>
    <path d="M80,50 C96,32 134,18 152,27"
          fill="none" stroke="rgba(255,255,255,0.22)" stroke-width="7" stroke-linecap="round"/>
  </svg>`,
};

// Eager-load all SVG images when the module is first imported.
// They'll be ready long before the user selects a prop (model loading takes longer).
const propImages = {};
for (const [name, svg] of Object.entries(PROP_SVGS)) {
  const img = new Image();
  img.src = `data:image/svg+xml,${encodeURIComponent(svg)}`;
  propImages[name] = img;
}

/**
 * Draw a prop sprite centred on its SVG anchor point, rotated to match face tilt.
 * @param {string}  name        — key in propImages
 * @param {number}  anchorX/Y   — canvas pixel where the anchor should land
 * @param {number}  angle       — face tilt in radians
 * @param {number}  drawWidth   — rendered width in canvas pixels
 * @param {number}  svgW/H      — SVG viewBox dimensions
 * @param {number}  axSvg/aySvg — anchor point inside the SVG
 */
function drawPropImage(ctx, name, anchorX, anchorY, angle, drawWidth, svgW, svgH, axSvg, aySvg) {
  const img = propImages[name];
  if (!img?.complete || !img.naturalWidth) return;
  const drawHeight = drawWidth * (svgH / svgW);
  ctx.save();
  ctx.translate(anchorX, anchorY);
  ctx.rotate(angle);
  ctx.drawImage(img, -(axSvg / svgW) * drawWidth, -(aySvg / svgH) * drawHeight, drawWidth, drawHeight);
  ctx.restore();
}

let faceLandmarker = null;
let animationId = null;
let lastVideoTime = -1;
let shared = null;
let uiSetup = false;

let currentMode = "landmarks";
let currentProp = "glasses";
let currentTrigger = "smile";
let capturedPhotos = [];
let photoStripActive = false;
let photoStripStep = 0;
const PHOTO_STRIP_PROMPTS = ["Neutral", "Smile!", "Silly face!", "Wink!"];

export async function activate(s) {
  shared = s;

  if (!faceLandmarker) {
    shared.statusEl.textContent = "Loading Face Mesh (MediaPipe)...";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  if (!uiSetup) {
    setupUI();
    uiSetup = true;
  }

  shared.overlay.textContent = "";
  shared.overlay.classList.remove("hidden");
  shared.statusEl.textContent = "Face mesh active";
  lastVideoTime = -1;
  detect();
}

export function deactivate() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  photoStripActive = false;
  shared = null;
}

function setupUI() {
  document.querySelectorAll("#mesh-modes .mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });
  document.querySelectorAll(".prop-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".prop-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      currentProp = btn.dataset.prop;
    });
  });
  document.querySelector(".prop-btn[data-prop='glasses']")?.classList.add("selected");
  document.querySelectorAll(".trigger-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".trigger-btn").forEach((b) => b.classList.remove("selected"));
      btn.classList.add("selected");
      currentTrigger = btn.dataset.trigger;
    });
  });
  document.querySelector(".trigger-btn[data-trigger='smile']")?.classList.add("selected");
  document.getElementById("btn-download-strip")?.addEventListener("click", downloadPhotoStrip);
  document.getElementById("btn-start-strip")?.addEventListener("click", startPhotoStrip);
}

function isShowLandmarksEnabled() {
  return document.getElementById("show-landmarks")?.checked ?? false;
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll("#mesh-modes .mode-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode)
  );
  document.querySelectorAll('[data-feature="face-mesh"] .submenu').forEach((s) =>
    s.classList.add("hidden")
  );
  const sub = document.getElementById(`submenu-${mode.replace(" ", "-")}`);
  if (sub) sub.classList.remove("hidden");

  if (shared) {
    if (mode === "photo-strip") {
      shared.photoStripPreview?.classList.remove("hidden");
    } else {
      shared.photoStripPreview?.classList.add("hidden");
      photoStripActive = false;
    }
  }
}

/**
 * Convert a normalized landmark to mirrored pixel coords.
 * MediaPipe landmarks are 0-1 in camera space; we flip x so they match
 * the mirrored video draw.
 */
function getKp(landmarks, i) {
  const lm = landmarks?.[i];
  if (!lm || !shared) return null;
  const w = shared.canvas.width;
  const h = shared.canvas.height;
  return { x: (1 - lm.x) * w, y: lm.y * h, z: lm.z ?? 0 };
}

function drawMeshConnections(ctx, landmarks, w, h) {
  if (!landmarks?.length) return;
  ctx.strokeStyle = "rgba(0, 255, 100, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const { start, end } of FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
    if (start >= landmarks.length || end >= landmarks.length) continue;
    ctx.moveTo((1 - landmarks[start].x) * w, landmarks[start].y * h);
    ctx.lineTo((1 - landmarks[end].x) * w, landmarks[end].y * h);
  }
  ctx.stroke();
}

function drawMeshDots(ctx, landmarks, w, h) {
  if (!landmarks?.length) return;
  ctx.fillStyle = "rgba(0, 255, 120, 1)";
  ctx.beginPath();
  for (const lm of landmarks) {
    ctx.moveTo((1 - lm.x) * w, lm.y * h);
    ctx.arc((1 - lm.x) * w, lm.y * h, 1.5, 0, Math.PI * 2);
  }
  ctx.fill();
}

function drawLandmarksVisualization(ctx, landmarks, w, h) {
  if (!landmarks?.length) return;
  drawMeshConnections(ctx, landmarks, w, h);
  drawMeshDots(ctx, landmarks, w, h);
}

function drawARProp(ctx, canvas, landmarks, prop) {
  const le = getKp(landmarks, L.leftEyeOuter);   // visual RIGHT after mirror
  const ro = getKp(landmarks, L.rightEyeOuter);  // visual LEFT after mirror
  const ft = getKp(landmarks, L.forehead);
  if (!le || !ro) return;

  // Inter-ocular distance (outer corner to outer corner) drives all sizing.
  const eyeSpan  = Math.hypot(le.x - ro.x, le.y - ro.y);
  const eyeCx    = (le.x + ro.x) / 2;
  const eyeCy    = (le.y + ro.y) / 2;
  // Rotation angle of the eye line — props tilt with the face.
  const angle    = Math.atan2(le.y - ro.y, le.x - ro.x);

  switch (prop) {
    case "glasses":
      // SVG 220×90, anchor (110, 44) = nose-bridge centre
      drawPropImage(ctx, "glasses", eyeCx, eyeCy, angle, eyeSpan * 2.05, 220, 90, 110, 44);
      break;

    case "sunglasses":
      // SVG 220×88, anchor (110, 40) = nose-bridge centre
      drawPropImage(ctx, "sunglasses", eyeCx, eyeCy, angle, eyeSpan * 2.05, 220, 88, 110, 40);
      break;

    case "hat":
      if (ft) {
        // SVG 240×185, anchor (120, 183) = bottom of brim
        // Place brim bottom just above the forehead landmark.
        drawPropImage(ctx, "hat", eyeCx, ft.y - eyeSpan * 0.05, angle, eyeSpan * 2.7, 240, 185, 120, 183);
      }
      break;

    case "crown":
      if (ft) {
        // SVG 200×130, anchor (100, 126) = bottom of base band
        drawPropImage(ctx, "crown", eyeCx, ft.y - eyeSpan * 0.05, angle, eyeSpan * 2.4, 200, 130, 100, 126);
      }
      break;

    case "mustache": {
      const nt = getKp(landmarks, L.noseTip);
      const ul = getKp(landmarks, L.upperLip);
      if (nt && ul) {
        // Anchor at SVG top-centre (90, 10); mustache hangs down from just below the nose.
        const mx = nt.x * 0.5 + ul.x * 0.5;
        const my = nt.y * 0.4 + ul.y * 0.6;
        drawPropImage(ctx, "mustache", mx, my, angle, eyeSpan * 1.55, 180, 72, 90, 10);
      }
      break;
    }

    case "bow":
      if (ft) {
        // SVG 160×100, anchor (80, 50) = bow centre
        // Sit above the forehead, roughly at the hairline.
        drawPropImage(ctx, "bow", eyeCx, ft.y - eyeSpan * 1.1, angle, eyeSpan * 1.9, 160, 100, 80, 50);
      }
      break;
  }
}

function detectSmile(landmarks) {
  const lm = getKp(landmarks, L.leftMouth);
  const rm = getKp(landmarks, L.rightMouth);
  const ul = getKp(landmarks, L.upperLip);
  if (!lm || !rm || !ul) return false;
  return (lm.y + rm.y) / 2 < ul.y + 5;
}

function getEyeOpenness(landmarks, outer, inner, upperLid, lowerLid) {
  const o = getKp(landmarks, outer);
  const i = getKp(landmarks, inner);
  const ul = getKp(landmarks, upperLid);
  const ll = getKp(landmarks, lowerLid);
  if (!o || !i || !ul || !ll) return 0.5;
  const eyeW = Math.hypot(o.x - i.x, o.y - i.y);
  return Math.min(1, Math.max(0, Math.hypot(ul.x - ll.x, ul.y - ll.y) / (eyeW || 1)));
}

function detectWink(landmarks) {
  const leftOpen = getEyeOpenness(landmarks, L.leftEyeOuter, L.leftEyeInner, 159, 145);
  const rightOpen = getEyeOpenness(landmarks, L.rightEyeOuter, L.rightEyeInner, 386, 374);
  return (leftOpen < 0.2 && rightOpen > 0.25) || (rightOpen < 0.2 && leftOpen > 0.25);
}

function detectSurprise(landmarks) {
  const lm = getKp(landmarks, L.leftMouth);
  const rm = getKp(landmarks, L.rightMouth);
  const ul = getKp(landmarks, L.upperLip);
  if (!lm || !rm || !ul) return false;
  const mouthOpen = Math.hypot(rm.x - lm.x, rm.y - lm.y);
  return mouthOpen > 25 && (lm.y + rm.y) / 2 > ul.y + 5;
}

function getExpression(landmarks) {
  if (detectSmile(landmarks)) return "Smile";
  if (detectWink(landmarks)) return "Wink";
  if (detectSurprise(landmarks)) return "Surprise";
  return "Neutral";
}

function getHeadPose(landmarks) {
  const lf = getKp(landmarks, L.leftFace);
  const rf = getKp(landmarks, L.rightFace);
  const n = getKp(landmarks, L.noseTip);
  if (!lf || !rf || !n) return "center";
  const diff = n.x - (lf.x + rf.x) / 2;
  if (diff < -15) return "left";
  if (diff > 15) return "right";
  return "center";
}

let lastCaptureTime = 0;
const CAPTURE_COOLDOWN = 1500;

function checkSmartCapture(landmarks) {
  const now = Date.now();
  if (now - lastCaptureTime < CAPTURE_COOLDOWN) return;
  let triggered = false;
  if (currentTrigger === "smile" && detectSmile(landmarks)) triggered = true;
  if (currentTrigger === "wink" && detectWink(landmarks)) triggered = true;
  if (currentTrigger === "surprise" && detectSurprise(landmarks)) triggered = true;
  if (triggered) {
    lastCaptureTime = now;
    capturePhoto();
  }
}

function applyExpressionFilter(expression) {
  const filters = {
    Smile: "sepia(0.3) saturate(1.2)",
    Wink: "hue-rotate(20deg)",
    Surprise: "contrast(1.1) saturate(1.3)",
    Neutral: "none",
  };
  return filters[expression] || "none";
}

function apply3DEffect(ctx, canvas, landmarks) {
  const n = getKp(landmarks, L.noseTip);
  const lf = getKp(landmarks, L.leftFace);
  const rf = getKp(landmarks, L.rightFace);
  if (!n || !lf || !rf) return;
  const tilt = (n.x - (lf.x + rf.x) / 2) / 50;
  const lm = landmarks[L.noseTip];
  const vignette = Math.max(0.3, 1 - Math.abs(lm?.z ?? 0) / 200);
  ctx.save();
  ctx.globalAlpha = vignette;
  ctx.fillStyle = `rgba(0,0,0,${0.4 - Math.abs(tilt) * 0.2})`;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function capturePhoto() {
  if (!shared) return;
  const c = document.createElement("canvas");
  c.width = shared.canvas.width;
  c.height = shared.canvas.height;
  const cctx = c.getContext("2d");
  cctx.drawImage(shared.canvas, 0, 0);
  capturedPhotos.push(c);
}

function startPhotoStrip() {
  photoStripActive = true;
  photoStripStep = 0;
  capturedPhotos = [];
  if (shared?.photoStripCanvases) shared.photoStripCanvases.innerHTML = "";
}

let photoStripCooldown = 0;
function updatePhotoStrip(landmarks) {
  if (!photoStripActive || !landmarks || !shared) return;
  const now = Date.now();
  if (now < photoStripCooldown) return;
  const prompt = PHOTO_STRIP_PROMPTS[photoStripStep];
  shared.posePrompt.textContent = prompt;
  shared.posePrompt.classList.remove("hidden");
  let ready = false;
  if (photoStripStep === 0) ready = getExpression(landmarks) === "Neutral";
  if (photoStripStep === 1) ready = detectSmile(landmarks);
  if (photoStripStep === 2) ready = detectSurprise(landmarks);
  if (photoStripStep === 3) ready = detectWink(landmarks);
  if (ready) {
    capturePhoto();
    photoStripStep++;
    photoStripCooldown = now + 800;
    if (photoStripStep >= PHOTO_STRIP_PROMPTS.length) {
      photoStripActive = false;
      shared.posePrompt.textContent = "Done!";
      renderPhotoStrip();
    }
  }
}

function renderPhotoStrip() {
  if (!shared?.photoStripCanvases) return;
  shared.photoStripCanvases.innerHTML = "";
  capturedPhotos.forEach((c) => {
    const thumb = document.createElement("canvas");
    thumb.width = 120;
    thumb.height = 90;
    thumb.getContext("2d").drawImage(c, 0, 0, 120, 90);
    shared.photoStripCanvases.appendChild(thumb);
  });
}

function downloadPhotoStrip() {
  if (capturedPhotos.length === 0) return;
  const stripW = 120 * capturedPhotos.length + 20;
  const strip = document.createElement("canvas");
  strip.width = stripW;
  strip.height = 110;
  const sctx = strip.getContext("2d");
  sctx.fillStyle = "#fff";
  sctx.fillRect(0, 0, stripW, 110);
  capturedPhotos.forEach((c, i) => {
    sctx.drawImage(c, 0, 0, c.width, c.height, 10 + i * 120, 10, 120, 90);
  });
  const a = document.createElement("a");
  a.download = "photobooth-strip.png";
  a.href = strip.toDataURL("image/png");
  a.click();
}

function detect() {
  if (!shared) return;
  if (!faceLandmarker || shared.video.readyState < 2 || shared.video.videoWidth === 0) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const { video, canvas, ctx, overlay, posePrompt } = shared;
  const w = canvas.width;
  const h = canvas.height;

  const now = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, now);

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    const landmarks = results.faceLandmarks?.[0];

    if (landmarks?.length > 0) {
      overlay.textContent = `${landmarks.length} landmarks`;

      const showLandmarks = currentMode === "landmarks" || isShowLandmarksEnabled();

      ctx.save();
      switch (currentMode) {
        case "landmarks":
          drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        case "ar-props":
          drawARProp(ctx, canvas, landmarks, currentProp);
          if (showLandmarks) drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        case "smart-capture":
          checkSmartCapture(landmarks);
          overlay.textContent = `Trigger: ${currentTrigger}`;
          if (showLandmarks) drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        case "expression": {
          const expr = getExpression(landmarks);
          overlay.textContent = expr;
          ctx.filter = applyExpressionFilter(expr);
          ctx.translate(w, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(video, 0, 0, w, h);
          ctx.filter = "none";
          if (showLandmarks) drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        }
        case "pose-guidance": {
          const pose = getHeadPose(landmarks);
          const prompts = { left: "Look left", right: "Look right", center: "Center" };
          posePrompt.textContent = prompts[pose];
          posePrompt.classList.remove("hidden");
          overlay.textContent = pose === "center" ? "Good!" : prompts[pose];
          if (showLandmarks) drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        }
        case "photo-strip":
          updatePhotoStrip(landmarks);
          if (showLandmarks) drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        case "3d-effects":
          apply3DEffect(ctx, canvas, landmarks);
          overlay.textContent = "3D lighting";
          if (showLandmarks) drawLandmarksVisualization(ctx, landmarks, w, h);
          break;
        default:
          drawLandmarksVisualization(ctx, landmarks, w, h);
      }
      ctx.restore();
    } else {
      overlay.textContent = "No face";
      posePrompt.classList.add("hidden");
    }
  }

  animationId = requestAnimationFrame(detect);
}
