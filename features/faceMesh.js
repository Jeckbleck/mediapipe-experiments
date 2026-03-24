import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";

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

const MESH_CONNECTIONS = [
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159], [159, 158], [158, 157], [157, 173], [173, 133],
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362],
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291], [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270], [270, 409], [409, 291],
  [276, 283], [283, 282], [282, 295], [295, 285], [300, 293], [293, 334], [334, 296], [296, 336],
  [46, 53], [53, 52], [52, 65], [65, 55], [70, 63], [63, 105], [105, 66], [66, 107],
];

let detector = null;
let animationId = null;
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

  if (!detector) {
    shared.statusEl.textContent = "Loading Face Mesh (TensorFlow)...";
    detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: "mediapipe",
        solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
        maxFaces: 1,
        refineLandmarks: false,
      }
    );
  }

  if (!uiSetup) {
    setupUI();
    uiSetup = true;
  }

  shared.overlay.textContent = "";
  shared.overlay.classList.remove("hidden");
  shared.statusEl.textContent = "Face mesh active";
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

function getKp(face, i) {
  const kp = face.keypoints?.[i];
  return kp ? { x: kp.x, y: kp.y, z: kp.z ?? 0 } : null;
}

function drawMeshConnections(ctx, keypoints) {
  if (!keypoints?.length) return;
  ctx.strokeStyle = "rgba(0, 255, 100, 1)";
  ctx.lineWidth = 2.5;
  for (const [a, b] of MESH_CONNECTIONS) {
    if (a < keypoints.length && b < keypoints.length) {
      const pa = keypoints[a];
      const pb = keypoints[b];
      if (pa && pb) {
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }
    }
  }
}

function drawMeshDots(ctx, keypoints) {
  if (!keypoints?.length) return;
  ctx.fillStyle = "rgba(0, 255, 120, 1)";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
  ctx.lineWidth = 1;
  for (const kp of keypoints) {
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawLandmarksVisualization(ctx, keypoints) {
  if (!keypoints?.length) return;
  drawMeshConnections(ctx, keypoints);
  drawMeshDots(ctx, keypoints);
}

function drawARProp(ctx, canvas, face, prop) {
  const le = getKp(face, L.leftEyeOuter);
  const ri = getKp(face, L.rightEyeInner);
  const li = getKp(face, L.leftEyeInner);
  const ro = getKp(face, L.rightEyeOuter);
  const nb = getKp(face, L.noseBridge);
  const ft = getKp(face, L.forehead);
  const ch = getKp(face, L.chin);
  if (!le || !ri || !nb) return;

  const eyeWidth = Math.hypot(ro.x - le.x, ro.y - le.y) * 1.4;
  const eyeCenterY = (le.y + ro.y) / 2;
  const scale = eyeWidth / 40;

  ctx.save();
  ctx.lineWidth = Math.max(2, scale * 0.5);
  ctx.strokeStyle = "#333";
  ctx.fillStyle = "#1a1a1a";

  switch (prop) {
    case "glasses":
      ctx.strokeRect(le.x - eyeWidth * 0.6, eyeCenterY - eyeWidth * 0.35, eyeWidth * 0.6, eyeWidth * 0.5);
      ctx.strokeRect(ro.x, eyeCenterY - eyeWidth * 0.35, eyeWidth * 0.6, eyeWidth * 0.5);
      ctx.beginPath();
      ctx.moveTo(le.x + eyeWidth * 0.55, eyeCenterY - eyeWidth * 0.1);
      ctx.lineTo(ro.x - eyeWidth * 0.15, eyeCenterY - eyeWidth * 0.1);
      ctx.stroke();
      if (nb) { ctx.beginPath(); ctx.arc(nb.x, nb.y, 4, 0, Math.PI * 2); ctx.fill(); }
      break;
    case "sunglasses":
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(le.x - eyeWidth * 0.65, eyeCenterY - eyeWidth * 0.4, eyeWidth * 0.65, eyeWidth * 0.6);
      ctx.fillRect(ro.x, eyeCenterY - eyeWidth * 0.4, eyeWidth * 0.65, eyeWidth * 0.6);
      ctx.strokeRect(le.x - eyeWidth * 0.65, eyeCenterY - eyeWidth * 0.4, eyeWidth * 0.65, eyeWidth * 0.6);
      ctx.strokeRect(ro.x, eyeCenterY - eyeWidth * 0.4, eyeWidth * 0.65, eyeWidth * 0.6);
      ctx.beginPath();
      ctx.moveTo(le.x + eyeWidth * 0.6, eyeCenterY - eyeWidth * 0.1);
      ctx.lineTo(ro.x - eyeWidth * 0.2, eyeCenterY - eyeWidth * 0.1);
      ctx.stroke();
      break;
    case "hat":
      if (ft && ch) {
        const hatY = ft.y - eyeWidth * 0.8;
        const hatW = eyeWidth * 2.2;
        const cx = (le.x + ro.x) / 2;
        ctx.fillStyle = "#8B4513";
        ctx.beginPath();
        ctx.ellipse(cx, hatY, hatW / 2, eyeWidth * 0.4, 0, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#654321";
        ctx.fillRect(cx - hatW / 2 - 5, hatY - eyeWidth * 0.35, hatW + 10, eyeWidth * 0.5);
        ctx.strokeRect(cx - hatW / 2 - 5, hatY - eyeWidth * 0.35, hatW + 10, eyeWidth * 0.5);
      }
      break;
    case "crown":
      if (ft) {
        const cx = (le.x + ro.x) / 2;
        const cy = ft.y - eyeWidth * 0.9;
        const pts = 5;
        ctx.fillStyle = "#FFD700";
        ctx.strokeStyle = "#B8860B";
        ctx.beginPath();
        ctx.moveTo(cx - eyeWidth * 1.1, cy + eyeWidth * 0.3);
        for (let i = 0; i <= pts; i++) {
          const a = (i / pts) * Math.PI * 0.8 + Math.PI * 0.1;
          ctx.lineTo(cx + Math.cos(a) * eyeWidth * 1.1, cy + Math.sin(a) * eyeWidth * 0.5);
        }
        ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      break;
    case "mustache": {
      const lm = getKp(face, L.leftMouth);
      const rm = getKp(face, L.rightMouth);
      const ul = getKp(face, L.upperLip);
      if (lm && rm && ul) {
        const my = (ul.y + (lm.y + rm.y) / 2) / 2;
        ctx.strokeStyle = "#333";
        ctx.lineWidth = Math.max(3, scale);
        ctx.beginPath();
        ctx.moveTo(lm.x, my);
        ctx.quadraticCurveTo((lm.x + rm.x) / 2, my + eyeWidth * 0.2, rm.x, my);
        ctx.stroke();
      }
      break;
    }
    case "bow":
      if (ft) {
        const bx = (le.x + ro.x) / 2;
        const by = ft.y - eyeWidth * 0.5;
        ctx.fillStyle = "#e91e63";
        ctx.beginPath();
        ctx.ellipse(bx - eyeWidth * 0.25, by, eyeWidth * 0.2, eyeWidth * 0.35, 0.3, 0, Math.PI * 2);
        ctx.ellipse(bx + eyeWidth * 0.25, by, eyeWidth * 0.2, eyeWidth * 0.35, -0.3, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#c2185b";
        ctx.beginPath();
        ctx.arc(bx, by, eyeWidth * 0.08, 0, Math.PI * 2);
        ctx.fill();
      }
      break;
  }
  ctx.restore();
}

function detectSmile(face) {
  const lm = getKp(face, L.leftMouth);
  const rm = getKp(face, L.rightMouth);
  const ul = getKp(face, L.upperLip);
  if (!lm || !rm || !ul) return false;
  return (lm.y + rm.y) / 2 < ul.y + 5;
}

function getEyeOpenness(face, outer, inner, upperLid, lowerLid) {
  const o = getKp(face, outer);
  const i = getKp(face, inner);
  const ul = getKp(face, upperLid);
  const ll = getKp(face, lowerLid);
  if (!o || !i || !ul || !ll) return 0.5;
  const eyeW = Math.hypot(o.x - i.x, o.y - i.y);
  return Math.min(1, Math.max(0, Math.hypot(ul.x - ll.x, ul.y - ll.y) / (eyeW || 1)));
}

function detectWink(face) {
  const leftOpen = getEyeOpenness(face, L.leftEyeOuter, L.leftEyeInner, 159, 145);
  const rightOpen = getEyeOpenness(face, L.rightEyeOuter, L.rightEyeInner, 386, 374);
  return (leftOpen < 0.2 && rightOpen > 0.25) || (rightOpen < 0.2 && leftOpen > 0.25);
}

function detectSurprise(face) {
  const lm = getKp(face, L.leftMouth);
  const rm = getKp(face, L.rightMouth);
  const ul = getKp(face, L.upperLip);
  if (!lm || !rm || !ul) return false;
  const mouthOpen = Math.hypot(rm.x - lm.x, rm.y - lm.y);
  return mouthOpen > 25 && (lm.y + rm.y) / 2 > ul.y + 5;
}

function getExpression(face) {
  if (detectSmile(face)) return "Smile";
  if (detectWink(face)) return "Wink";
  if (detectSurprise(face)) return "Surprise";
  return "Neutral";
}

function getHeadPose(face) {
  const lf = getKp(face, L.leftFace);
  const rf = getKp(face, L.rightFace);
  const n = getKp(face, L.noseTip);
  if (!lf || !rf || !n) return "center";
  const diff = n.x - (lf.x + rf.x) / 2;
  if (diff < -15) return "left";
  if (diff > 15) return "right";
  return "center";
}

let lastCaptureTime = 0;
const CAPTURE_COOLDOWN = 1500;

function checkSmartCapture(face) {
  const now = Date.now();
  if (now - lastCaptureTime < CAPTURE_COOLDOWN) return;
  let triggered = false;
  if (currentTrigger === "smile" && detectSmile(face)) triggered = true;
  if (currentTrigger === "wink" && detectWink(face)) triggered = true;
  if (currentTrigger === "surprise" && detectSurprise(face)) triggered = true;
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

function apply3DEffect(ctx, canvas, face) {
  const n = getKp(face, L.noseTip);
  const lf = getKp(face, L.leftFace);
  const rf = getKp(face, L.rightFace);
  if (!n || !lf || !rf) return;
  const tilt = (n.x - (lf.x + rf.x) / 2) / 50;
  const vignette = Math.max(0.3, 1 - Math.abs(n.z || 0) / 200);
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
function updatePhotoStrip(face) {
  if (!photoStripActive || !face || !shared) return;
  const now = Date.now();
  if (now < photoStripCooldown) return;
  const prompt = PHOTO_STRIP_PROMPTS[photoStripStep];
  shared.posePrompt.textContent = prompt;
  shared.posePrompt.classList.remove("hidden");
  let ready = false;
  if (photoStripStep === 0) ready = getExpression(face) === "Neutral";
  if (photoStripStep === 1) ready = detectSmile(face);
  if (photoStripStep === 2) ready = detectSurprise(face);
  if (photoStripStep === 3) ready = detectWink(face);
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

async function detect() {
  if (!shared) return;
  if (!detector || shared.video.readyState < 2 || shared.video.videoWidth === 0) {
    animationId = requestAnimationFrame(detect);
    return;
  }

  const { video, canvas, ctx, overlay, posePrompt } = shared;
  const faces = await detector.estimateFaces(video, { flipHorizontal: true });

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0);
  ctx.restore();

  if (faces.length > 0) {
    const face = faces[0];
    overlay.textContent = `${face.keypoints?.length ?? 0} landmarks`;

    const showLandmarks = currentMode === "landmarks" || isShowLandmarksEnabled();

    ctx.save();
    switch (currentMode) {
      case "landmarks":
        drawLandmarksVisualization(ctx, face.keypoints);
        break;
      case "ar-props":
        drawARProp(ctx, canvas, face, currentProp);
        if (showLandmarks) drawLandmarksVisualization(ctx, face.keypoints);
        break;
      case "smart-capture":
        checkSmartCapture(face);
        overlay.textContent = `Trigger: ${currentTrigger}`;
        if (showLandmarks) drawLandmarksVisualization(ctx, face.keypoints);
        break;
      case "expression": {
        const expr = getExpression(face);
        overlay.textContent = expr;
        ctx.filter = applyExpressionFilter(expr);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.filter = "none";
        if (showLandmarks) drawLandmarksVisualization(ctx, face.keypoints);
        break;
      }
      case "pose-guidance": {
        const pose = getHeadPose(face);
        const prompts = { left: "Look left", right: "Look right", center: "Center" };
        posePrompt.textContent = prompts[pose];
        posePrompt.classList.remove("hidden");
        overlay.textContent = pose === "center" ? "Good!" : prompts[pose];
        if (showLandmarks) drawLandmarksVisualization(ctx, face.keypoints);
        break;
      }
      case "photo-strip":
        updatePhotoStrip(face);
        if (showLandmarks) drawLandmarksVisualization(ctx, face.keypoints);
        break;
      case "3d-effects":
        apply3DEffect(ctx, canvas, face);
        overlay.textContent = "3D lighting";
        if (showLandmarks) drawLandmarksVisualization(ctx, face.keypoints);
        break;
      default:
        drawLandmarksVisualization(ctx, face.keypoints);
    }
    ctx.restore();
  } else {
    overlay.textContent = "No face";
    posePrompt.classList.add("hidden");
  }

  animationId = requestAnimationFrame(detect);
}
