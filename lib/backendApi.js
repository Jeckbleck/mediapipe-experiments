/**
 * Client for the Python backend face processing API.
 * Use when backend is running (npm run backend) for more reliable face swap/AR.
 */

const API_BASE = "/api";

let _backendAvailable = null;

export async function isBackendAvailable(forceRefresh = false) {
  if (!forceRefresh && _backendAvailable !== null) return _backendAvailable;
  try {
    const r = await fetch(`${API_BASE}/health`);
    const data = await r.json();
    _backendAvailable = data?.backend === "python";
  } catch {
    _backendAvailable = false;
  }
  return _backendAvailable;
}

/** Reset cache so next isBackendAvailable() re-checks. Call when user toggles the checkbox. */
export function resetBackendCache() {
  _backendAvailable = null;
}

/**
 * Process a frame via the Python backend.
 * @param {HTMLCanvasElement} canvas - Canvas with the current frame
 * @param {Object} options
 * @param {string} options.mode - "landmarks" | "face-swap" | "ar-props" | "face-swap-multi"
 * @param {string} [options.prop] - For ar-props: "glasses" | "hat" | "crown" | "sunglasses" | "mustache" | "bow"
 * @param {HTMLImageElement} [options.referenceImage] - For face-swap: reference face image
 * @returns {Promise<{success: boolean, image?: string, error?: string}>}
 */
export async function processFrame(canvas, options = {}) {
  const { mode = "landmarks", prop = "glasses", referenceImage } = options;
  const frameB64 = canvas.toDataURL("image/jpeg", 0.85);
  const form = new FormData();
  form.append("frame", frameB64);
  form.append("mode", mode);
  form.append("prop", prop);
  if (referenceImage && mode === "face-swap") {
    const refCanvas = document.createElement("canvas");
    refCanvas.width = referenceImage.naturalWidth || referenceImage.width;
    refCanvas.height = referenceImage.naturalHeight || referenceImage.height;
    const refCtx = refCanvas.getContext("2d");
    refCtx.drawImage(referenceImage, 0, 0);
    form.append("reference_image", refCanvas.toDataURL("image/jpeg", 0.9));
  }
  const r = await fetch(`${API_BASE}/process`, {
    method: "POST",
    body: form,
  });
  const data = await r.json();
  return data;
}
