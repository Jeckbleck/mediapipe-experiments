import { HandLandmarker, ImageSegmenter, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH  = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const SEG_MODEL  = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";

const BG_CAT    = 0;
const MAX_SIGNS = 5;
const SAMPLE_N  = 6;
const COUNTDOWN = 3000;
const LS_KEY    = "customSigns_v1";

let handLandmarker = null;
let segmenter      = null;
let shared         = null;
let animId         = null;
let lastVideoTime  = -1;
let uiSetup        = false;

// signs[i]: { name: string, vector: number[] } | null
let signs      = Array(MAX_SIGNS).fill(null);
let bgDataUrls = Array(MAX_SIGNS).fill(null);
const bgImgs   = Array(MAX_SIGNS).fill(null);

let threshold    = 1.4;
let detectedIdx  = -1;
let detectedConf = 0;

// Recording state
let recSlot    = -1;
let recPhase   = "idle"; // "idle" | "countdown" | "sampling"
let recEnd     = 0;
let recSamples = [];

// Latest hand landmarks (kept for re-drawing when video frame unchanged)
let lastL = null;
let lastR = null;

// Segmentation offscreen canvases
const maskCanvas   = document.createElement("canvas");
const maskCtx      = maskCanvas.getContext("2d", { willReadFrequently: true });
const smoothCanvas = document.createElement("canvas");
const smoothCtx    = smoothCanvas.getContext("2d");
const personCanvas = document.createElement("canvas");
const personCtx    = personCanvas.getContext("2d");
let maskImageData  = null;
let pendingSeg     = null;

// ─── Persistence ──────────────────────────────────────────────────────────────

function loadStorage() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return;
    JSON.parse(raw).forEach((item, i) => {
      if (i >= MAX_SIGNS || !item) return;
      signs[i] = { name: item.name, vector: item.vector, mode: item.mode ?? "two" };
      if (item.bgDataUrl) {
        bgDataUrls[i] = item.bgDataUrl;
        const img = new Image();
        img.src = item.bgDataUrl;
        bgImgs[i] = img;
      }
    });
  } catch (_) {}
}

function saveStorage() {
  localStorage.setItem(LS_KEY, JSON.stringify(
    signs.map((s, i) => s
      ? { name: s.name, vector: s.vector, bgDataUrl: bgDataUrls[i] ?? null }
      : null
    )
  ));
}

// ─── Normalization ────────────────────────────────────────────────────────────
// Origin = midpoint of both wrists. Scale = wrist-to-wrist distance.
// This makes the vector invariant to position, rotation, and camera distance
// while preserving the relative geometry between the hands (incl. when touching).

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function buildVector(L, R) {
  const lw = L[0], rw = R[0];
  const ox = (lw.x + rw.x) / 2;
  const oy = (lw.y + rw.y) / 2;
  const sc = dist2D(lw, rw);
  if (sc < 0.02) return null;
  return [...L, ...R].flatMap(p => [(p.x - ox) / sc, (p.y - oy) / sc]);
}

function buildVectorSingle(lm) {
  const wrist  = lm[0];
  const midMCP = lm[9]; // middle-finger knuckle — stable scale reference
  const sc = dist2D(wrist, midMCP);
  if (sc < 0.01) return null;
  return lm.flatMap(p => [(p.x - wrist.x) / sc, (p.y - wrist.y) / sc]);
}

function vecDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

function avgVecs(vecs) {
  const out = new Array(vecs[0].length).fill(0);
  vecs.forEach(v => v.forEach((x, i) => { out[i] += x; }));
  return out.map(x => x / vecs.length);
}

// ─── Hand extraction ──────────────────────────────────────────────────────────

function getHands(results) {
  let L = null, R = null;
  results.landmarks?.forEach((lm, i) => {
    const label = results.handedness[i]?.[0]?.categoryName;
    if (label === "Left")  L = lm;
    if (label === "Right") R = lm;
  });
  return { L, R };
}

// ─── Sign matching ────────────────────────────────────────────────────────────

function matchSign(L, R) {
  let bestIdx = -1, bestDist = Infinity;

  signs.forEach((s, i) => {
    if (!s) return;
    let vec = null;
    if (s.mode === "two") {
      if (!L || !R) return;
      vec = buildVector(L, R);
    } else if (s.mode === "left") {
      if (!L) return;
      vec = buildVectorSingle(L);
    } else {
      if (!R) return;
      vec = buildVectorSingle(R);
    }
    if (!vec) return;
    const d = vecDist(vec, s.vector);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  });

  if (bestIdx !== -1 && bestDist < threshold) {
    detectedIdx  = bestIdx;
    detectedConf = Math.max(0, 1 - bestDist / threshold);
  } else {
    detectedIdx  = -1;
    detectedConf = 0;
  }
}

// ─── Recording ────────────────────────────────────────────────────────────────

function startRecord(slot) {
  recSlot    = slot;
  recPhase   = "countdown";
  recEnd     = performance.now() + COUNTDOWN;
  recSamples = [];
  updateSlotUI(slot);
}

function cancelRecord() {
  const prev = recSlot;
  recSlot  = -1;
  recPhase = "idle";
  recSamples = [];
  if (prev >= 0) updateSlotUI(prev);
}

function finishRecord() {
  const slot = recSlot;

  // Determine mode from the majority of captured samples
  const modes = recSamples.map(s => s.mode);
  const mode  = ["two", "left", "right"].sort(
    (a, b) => modes.filter(m => m === b).length - modes.filter(m => m === a).length
  )[0];

  const vec    = avgVecs(recSamples.filter(s => s.mode === mode).map(s => s.vec));
  const nameEl = document.querySelector(`.sign-name[data-slot="${slot}"]`);
  const name   = nameEl?.value.trim() || `Sign ${slot + 1}`;
  signs[slot]  = { name, vector: vec, mode };
  recSlot      = -1;
  recPhase     = "idle";
  recSamples   = [];
  saveStorage();
  updateSlotUI(slot);
  if (shared) shared.statusEl.textContent = `"${name}" saved (${mode === "two" ? "both hands" : mode + " hand"})`;
}

function tickRecord(L, R, now) {
  if (recPhase === "countdown" && now >= recEnd) {
    recPhase = "sampling";
  }
  if (recPhase === "sampling") {
    let vec = null, mode = null;
    if (L && R) {
      vec = buildVector(L, R);
      mode = "two";
    } else if (L) {
      vec = buildVectorSingle(L);
      mode = "left";
    } else if (R) {
      vec = buildVectorSingle(R);
      mode = "right";
    }
    if (vec) recSamples.push({ vec, mode });
    if (recSamples.length >= SAMPLE_N) finishRecord();
  }
}

// ─── UI ───────────────────────────────────────────────────────────────────────

function setupUI() {
  loadStorage();

  const container = document.getElementById("sign-slots");
  if (!container) return;
  container.innerHTML = "";

  for (let i = 0; i < MAX_SIGNS; i++) {
    const el = document.createElement("div");
    el.className = "sign-slot";
    el.innerHTML = `
      <div class="sign-slot-row">
        <span class="sign-slot-num">${i + 1}</span>
        <input type="text" class="sign-name" data-slot="${i}"
               placeholder="Sign name…" value="${signs[i]?.name ?? ""}">
        <button type="button" class="sign-delete" data-slot="${i}" title="Delete">✕</button>
      </div>
      <div class="sign-slot-row">
        <label class="btn-secondary sign-bg-label" style="cursor:pointer;font-size:0.78rem">
          BG image
          <input type="file" accept="image/*" hidden class="sign-bg-file" data-slot="${i}">
        </label>
        <button type="button" class="sign-record btn-primary" data-slot="${i}">Record</button>
        <span class="sign-status" data-slot="${i}">${signs[i] ? "✓" : "—"}</span>
      </div>
      <div class="sign-thumb" data-slot="${i}">${bgDataUrls[i] ? `<img src="${bgDataUrls[i]}" class="sign-bg-thumb">` : ""}</div>
    `;
    container.appendChild(el);

    el.querySelector(".sign-record").addEventListener("click", () =>
      recSlot === i ? cancelRecord() : startRecord(i)
    );
    el.querySelector(".sign-delete").addEventListener("click", () => deleteSign(i));
    el.querySelector(".sign-bg-file").addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        const img = new Image();
        img.onload = () => {
          // Resize to ≤640×360 JPEG before storing to stay within localStorage limits
          const maxW = 640, maxH = 360;
          const scale = Math.min(1, maxW / img.width, maxH / img.height);
          const rc = document.createElement("canvas");
          rc.width  = Math.round(img.width  * scale);
          rc.height = Math.round(img.height * scale);
          rc.getContext("2d").drawImage(img, 0, 0, rc.width, rc.height);
          const dataUrl = rc.toDataURL("image/jpeg", 0.85);
          bgDataUrls[i] = dataUrl;
          const loaded = new Image();
          loaded.src = dataUrl;
          bgImgs[i] = loaded;
          const thumb = document.querySelector(`.sign-thumb[data-slot="${i}"]`);
          if (thumb) thumb.innerHTML = `<img src="${dataUrl}" class="sign-bg-thumb">`;
          saveStorage();
        };
        img.src = ev.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  const slider  = document.getElementById("sign-sensitivity");
  const sliderV = document.getElementById("sign-sensitivity-val");
  slider?.addEventListener("input", () => {
    threshold = parseFloat(slider.value);
    if (sliderV) sliderV.textContent = threshold.toFixed(1);
  });
}

function updateSlotUI(slot) {
  const statusEl  = document.querySelector(`.sign-status[data-slot="${slot}"]`);
  const recordBtn = document.querySelector(`.sign-record[data-slot="${slot}"]`);
  if (!statusEl || !recordBtn) return;

  const active = recSlot === slot;
  recordBtn.textContent = active
    ? (recPhase === "countdown" ? "Cancel" : "Sampling…")
    : "Record";
  statusEl.textContent = !active && signs[slot] ? "✓" : "—";
}

function deleteSign(slot) {
  signs[slot]      = null;
  bgDataUrls[slot] = null;
  bgImgs[slot]     = null;
  const thumb = document.querySelector(`.sign-thumb[data-slot="${slot}"]`);
  if (thumb) thumb.innerHTML = "";
  const nameEl = document.querySelector(`.sign-name[data-slot="${slot}"]`);
  if (nameEl) nameEl.value = "";
  updateSlotUI(slot);
  saveStorage();
  if (detectedIdx === slot) { detectedIdx = -1; detectedConf = 0; }
}

// ─── Segmentation compositing ─────────────────────────────────────────────────

function applyBg(ctx, video, segResult, w, h) {
  const bgImg = detectedIdx >= 0 ? bgImgs[detectedIdx] : null;

  if (!segResult || !bgImg?.complete) {
    ctx.drawImage(video, 0, 0, w, h);
    return;
  }

  const mask = segResult.categoryMask;
  const mw = mask.width, mh = mask.height;
  const data = mask.getAsUint8Array();

  if (maskCanvas.width !== mw || maskCanvas.height !== mh) {
    maskCanvas.width  = mw;
    maskCanvas.height = mh;
    maskImageData = null;
  }
  if (!maskImageData) maskImageData = maskCtx.createImageData(mw, mh);

  const px = maskImageData.data;
  for (let i = 0; i < data.length; i++) {
    const j = i * 4;
    px[j] = px[j + 1] = px[j + 2] = 255;
    px[j + 3] = data[i] !== BG_CAT ? 255 : 0;
  }
  maskCtx.putImageData(maskImageData, 0, 0);

  if (smoothCanvas.width !== mw || smoothCanvas.height !== mh) {
    smoothCanvas.width = mw;
    smoothCanvas.height = mh;
  }
  smoothCtx.clearRect(0, 0, mw, mh);
  smoothCtx.filter = "blur(2px)";
  smoothCtx.drawImage(maskCanvas, 0, 0);
  smoothCtx.filter = "none";

  if (personCanvas.width !== w || personCanvas.height !== h) {
    personCanvas.width = w;
    personCanvas.height = h;
  }
  personCtx.clearRect(0, 0, w, h);
  personCtx.drawImage(video, 0, 0, w, h);
  personCtx.globalCompositeOperation = "destination-in";
  const sc = Math.min(mw / w, mh / h);
  const sw = Math.round(w * sc), sh = Math.round(h * sc);
  const ox = Math.round((mw - sw) / 2), oy = Math.round((mh - sh) / 2);
  personCtx.drawImage(smoothCanvas, ox, oy, sw, sh, 0, 0, w, h);
  personCtx.globalCompositeOperation = "source-over";

  ctx.drawImage(bgImg, 0, 0, w, h);
  ctx.drawImage(personCanvas, 0, 0, w, h);
}

// ─── Canvas overlays ──────────────────────────────────────────────────────────

function drawRecordOverlay(ctx, w, h) {
  const now = performance.now();
  if (recPhase === "countdown") {
    const remaining = Math.ceil((recEnd - now) / 1000);
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#e4e4e7";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 88px monospace";
    ctx.fillText(String(remaining), w / 2, h / 2);
    ctx.font = "18px monospace";
    ctx.fillStyle = "#a1a1aa";
    ctx.fillText("Hold both hands in position", w / 2, h / 2 + 64);
    ctx.restore();
  } else if (recPhase === "sampling") {
    const prog = recSamples.length / SAMPLE_N;
    ctx.save();
    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth   = 6;
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 44, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * prog);
    ctx.stroke();
    ctx.restore();
  }
}

function drawDetectionOverlay(ctx, w, h) {
  if (detectedIdx < 0) return;
  const name = signs[detectedIdx]?.name ?? "";
  const conf = Math.round(detectedConf * 100);

  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.fillRect(0, h - 56, w, 56);
  ctx.fillStyle = "#e4e4e7";
  ctx.font      = "bold 22px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(name, w / 2, h - 34);
  ctx.fillStyle = "#71717a";
  ctx.font      = "13px monospace";
  ctx.fillText(`${conf}% confidence`, w / 2, h - 14);
  ctx.restore();
}

function drawHandDots(ctx, L, R, w, h) {
  [[L, "#818cf8"], [R, "#34d399"]].forEach(([lm, color]) => {
    if (!lm) return;
    [4, 8, 12, 16, 20].forEach(idx => {
      ctx.save();
      ctx.fillStyle   = color;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 6;
      ctx.beginPath();
      ctx.arc(lm[idx].x * w, lm[idx].y * h, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  });
}

// ─── Activate / deactivate ───────────────────────────────────────────────────

function onVideoCanvasResize() {
  lastVideoTime = -1;
}

export async function activate(s) {
  shared = s;
  s.video.addEventListener("videocanvasresize", onVideoCanvasResize);

  if (!handLandmarker || !segmenter) {
    shared.statusEl.textContent = "Loading models…";
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
    if (!handLandmarker) {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
      });
    }
    if (!segmenter) {
      segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: SEG_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
    }
  }

  if (!uiSetup) { setupUI(); uiSetup = true; }

  shared.statusEl.textContent = "Show both hands to detect — or record a new sign";
  lastVideoTime = -1;
  detect();
}

export function deactivate() {
  if (animId) cancelAnimationFrame(animId);
  animId = null;
  shared?.video.removeEventListener("videocanvasresize", onVideoCanvasResize);
  shared = null;
  cancelRecord();
}

// ─── Detection loop ───────────────────────────────────────────────────────────

function detect() {
  if (!shared) return;
  const { video, canvas, ctx } = shared;
  const w = canvas.width, h = canvas.height;

  if (video.readyState >= 2 && lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();

    const handRes = handLandmarker.detectForVideo(video, now);
    const { L, R } = getHands(handRes);
    lastL = L; lastR = R;

    if (recPhase !== "idle") {
      tickRecord(L, R, now);
    } else {
      matchSign(L, R);
    }

    segmenter.segmentForVideo(video, now, result => { pendingSeg = result; });
  }

  applyBg(ctx, video, pendingSeg, w, h);
  drawHandDots(ctx, lastL, lastR, w, h);
  drawDetectionOverlay(ctx, w, h);
  drawRecordOverlay(ctx, w, h);

  animId = requestAnimationFrame(detect);
}
