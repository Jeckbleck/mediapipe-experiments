import { FaceLandmarker } from "@mediapipe/tasks-vision";

/**
 * MediaPipe face mesh triangles: derive from official tesselation edges so
 * topology matches the model (avoids Delaunay cutting across the face).
 */
function connectionsToTriangles(connections) {
  const adj = new Map();
  function addEdge(a, b) {
    if (a === b) return;
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a).add(b);
    adj.get(b).add(a);
  }
  for (const { start, end } of connections) {
    addEdge(start, end);
  }
  const seen = new Set();
  const triangles = [];
  for (const { start: u, end: v } of connections) {
    if (u === v) continue;
    const nu = adj.get(u);
    const nv = adj.get(v);
    if (!nu || !nv) continue;
    for (const w of nu) {
      if (w === u || w === v) continue;
      if (!nv.has(w)) continue;
      const a = Math.min(u, v, w);
      const c = Math.max(u, v, w);
      const b = u + v + w - a - c;
      const key = `${a},${b},${c}`;
      if (seen.has(key)) continue;
      seen.add(key);
      triangles.push([u, v, w]);
    }
  }
  return triangles;
}

const FACE_MESH_TRIANGLES = connectionsToTriangles(
  FaceLandmarker.FACE_LANDMARKS_TESSELATION
);

/**
 * Convert normalized landmarks (0-1) to pixel coordinates
 */
export function toPixelCoords(landmarks, width, height) {
  return landmarks.map((lm) => ({
    x: lm.x * width,
    y: lm.y * height,
  }));
}

function orientLandmarkTriangleCCW(landmarks, w, h, i0, i1, i2) {
  const p0 = landmarks[i0];
  const p1 = landmarks[i1];
  const p2 = landmarks[i2];
  if (!p0 || !p1 || !p2) return null;
  const x0 = p0.x * w;
  const y0 = p0.y * h;
  const x1 = p1.x * w;
  const y1 = p1.y * h;
  const x2 = p2.x * w;
  const y2 = p2.y * h;
  const cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  if (Math.abs(cross) < 1e-8) return null;
  if (cross < 0) return [i0, i2, i1];
  return [i0, i1, i2];
}

function triPixels(landmarks, w, h, i0, i1, i2) {
  return [
    { x: landmarks[i0].x * w, y: landmarks[i0].y * h },
    { x: landmarks[i1].x * w, y: landmarks[i1].y * h },
    { x: landmarks[i2].x * w, y: landmarks[i2].y * h },
  ];
}

/**
 * Barycentric coordinates: is point p inside triangle (a,b,c)?
 * Matches P = (1-u-v)*A + v*B + u*C with a=A, b=B, c=C (see warpTriangle).
 */
function barycentric(p, a, b, c) {
  const v0 = { x: c.x - a.x, y: c.y - a.y };
  const v1 = { x: b.x - a.x, y: b.y - a.y };
  const v2 = { x: p.x - a.x, y: p.y - a.y };
  const dot00 = v0.x * v0.x + v0.y * v0.y;
  const dot01 = v0.x * v1.x + v0.y * v1.y;
  const dot02 = v0.x * v2.x + v0.y * v2.y;
  const dot11 = v1.x * v1.x + v1.y * v1.y;
  const dot12 = v1.x * v2.x + v1.y * v2.y;
  const invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
  const u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  const v = (dot00 * dot12 - dot01 * dot02) * invDenom;
  const w = 1 - u - v;
  if (u >= 0 && v >= 0 && w >= 0) return { u, v, w };
  return null;
}

/**
 * Sample pixel from source image at (x, y) with bilinear interpolation
 */
function samplePixel(imageData, x, y, width, height) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  if (x0 < 0 || x0 >= width - 1 || y0 < 0 || y0 >= height - 1) {
    return [0, 0, 0, 0];
  }
  const idx = (y0 * width + x0) * 4;
  return [
    imageData.data[idx],
    imageData.data[idx + 1],
    imageData.data[idx + 2],
    imageData.data[idx + 3],
  ];
}

/**
 * Warp a triangle from source to destination.
 * Barycentric: P = w*A + v*B + u*C where dstTri[0]=A, [1]=B, [2]=C.
 */
function warpTriangle(srcData, dstData, srcTri, dstTri, srcW, srcH, dstW, dstH) {
  const minX = Math.max(0, Math.floor(Math.min(dstTri[0].x, dstTri[1].x, dstTri[2].x)));
  const maxX = Math.min(dstW, Math.ceil(Math.max(dstTri[0].x, dstTri[1].x, dstTri[2].x)) + 1);
  const minY = Math.max(0, Math.floor(Math.min(dstTri[0].y, dstTri[1].y, dstTri[2].y)));
  const maxY = Math.min(dstH, Math.ceil(Math.max(dstTri[0].y, dstTri[1].y, dstTri[2].y)) + 1);

  for (let y = minY; y < maxY; y++) {
    for (let x = minX; x < maxX; x++) {
      const p = { x, y };
      const bc = barycentric(p, dstTri[0], dstTri[1], dstTri[2]);
      if (bc) {
        const srcX = bc.w * srcTri[0].x + bc.v * srcTri[1].x + bc.u * srcTri[2].x;
        const srcY = bc.w * srcTri[0].y + bc.v * srcTri[1].y + bc.u * srcTri[2].y;
        const [r, g, b, a] = samplePixel(srcData, srcX, srcY, srcW, srcH);
        const idx = (y * dstW + x) * 4;
        dstData.data[idx] = r;
        dstData.data[idx + 1] = g;
        dstData.data[idx + 2] = b;
        dstData.data[idx + 3] = a;
      }
    }
  }
}

/**
 * Swap faces between two sets of landmarks (same resolution).
 */
export function swapFaces(ctx, srcImg, landmarksA, landmarksB, width, height) {
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = width;
  srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.drawImage(srcImg, 0, 0);
  const srcData = srcCtx.getImageData(0, 0, width, height);

  const dstCanvas = document.createElement("canvas");
  dstCanvas.width = width;
  dstCanvas.height = height;
  const dstCtx = dstCanvas.getContext("2d");
  dstCtx.drawImage(srcImg, 0, 0);
  let dstData = dstCtx.getImageData(0, 0, width, height);

  const nA = landmarksA.length;
  const nB = landmarksB.length;

  for (const raw of FACE_MESH_TRIANGLES) {
    let [i0, i1, i2] = raw;
    if (i0 >= nA || i1 >= nA || i2 >= nA || i0 >= nB || i1 >= nB || i2 >= nB) continue;
    const oriented = orientLandmarkTriangleCCW(landmarksA, width, height, i0, i1, i2);
    if (!oriented) continue;
    [i0, i1, i2] = oriented;
    const dstTri = triPixels(landmarksA, width, height, i0, i1, i2);
    const srcTri = triPixels(landmarksB, width, height, i0, i1, i2);
    warpTriangle(srcData, dstData, srcTri, dstTri, width, height, width, height);
  }

  dstCtx.putImageData(dstData, 0, 0);

  for (const raw of FACE_MESH_TRIANGLES) {
    let [i0, i1, i2] = raw;
    if (i0 >= nA || i1 >= nA || i2 >= nA || i0 >= nB || i1 >= nB || i2 >= nB) continue;
    const oriented = orientLandmarkTriangleCCW(landmarksB, width, height, i0, i1, i2);
    if (!oriented) continue;
    [i0, i1, i2] = oriented;
    const srcTri = triPixels(landmarksA, width, height, i0, i1, i2);
    const dstTri = triPixels(landmarksB, width, height, i0, i1, i2);
    warpTriangle(srcData, dstData, srcTri, dstTri, width, height, width, height);
  }

  dstCtx.putImageData(dstData, 0, 0);
  ctx.drawImage(dstCanvas, 0, 0);
}

/**
 * Warp reference face texture onto the user's face using MediaPipe mesh triangles.
 */
export function warpReferenceFaceOntoTarget(
  ctx,
  refImg,
  refLandmarks,
  targetLandmarks,
  width,
  height
) {
  const srcW = refImg.naturalWidth || refImg.videoWidth || refImg.width || 0;
  const srcH = refImg.naturalHeight || refImg.videoHeight || refImg.height || 0;
  if (srcW < 1 || srcH < 1 || width < 1 || height < 1) return;

  const refCanvas = document.createElement("canvas");
  refCanvas.width = srcW;
  refCanvas.height = srcH;
  const refCtx = refCanvas.getContext("2d");
  refCtx.drawImage(refImg, 0, 0);
  const srcData = refCtx.getImageData(0, 0, srcW, srcH);

  const dstData = ctx.getImageData(0, 0, width, height);

  const nRef = refLandmarks.length;
  const nTgt = targetLandmarks.length;

  for (const raw of FACE_MESH_TRIANGLES) {
    let [i0, i1, i2] = raw;
    if (i0 >= nRef || i1 >= nRef || i2 >= nRef || i0 >= nTgt || i1 >= nTgt || i2 >= nTgt) {
      continue;
    }
    const oriented = orientLandmarkTriangleCCW(targetLandmarks, width, height, i0, i1, i2);
    if (!oriented) continue;
    [i0, i1, i2] = oriented;
    const dstTri = triPixels(targetLandmarks, width, height, i0, i1, i2);
    const srcTri = triPixels(refLandmarks, srcW, srcH, i0, i1, i2);
    warpTriangle(srcData, dstData, srcTri, dstTri, srcW, srcH, width, height);
  }

  ctx.putImageData(dstData, 0, 0);
}

/**
 * Get face bounding box from landmarks (with padding)
 */
function getFaceBounds(landmarks, width, height, padding = 0.25) {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const lm of landmarks) {
    minX = Math.min(minX, lm.x);
    minY = Math.min(minY, lm.y);
    maxX = Math.max(maxX, lm.x);
    maxY = Math.max(maxY, lm.y);
  }
  const w = (maxX - minX) * (1 + padding * 2);
  const h = (maxY - minY) * (1 + padding * 2);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const x = Math.max(0, (cx - w / 2) * width);
  const y = Math.max(0, (cy - h / 2) * height);
  const sw = Math.min(width - x, w * width);
  const sh = Math.min(height - y, h * height);
  return { x, y, w: sw, h: sh };
}

/**
 * Overlay source face onto target face in the image
 * Uses simple bounding-box warp (reliable)
 */
export function overlayFace(ctx, srcImg, srcLandmarks, dstLandmarks, width, height) {
  const log = (msg, data) => {
    if (typeof window !== "undefined" && window.DEBUG_FACESWAP) {
      console.log("[FaceSwap:overlay]", msg, data ?? "");
    }
  };

  const srcW = srcImg.naturalWidth || srcImg.videoWidth || srcImg.width || width;
  const srcH = srcImg.naturalHeight || srcImg.videoHeight || srcImg.height || height;

  log("overlayFace called", { srcW, srcH, width, height, srcLandmarksCount: srcLandmarks?.length, dstLandmarksCount: dstLandmarks?.length });

  if (srcW === 0 || srcH === 0) {
    log("SKIP: zero source dimensions");
    return;
  }

  const srcBounds = getFaceBounds(srcLandmarks, srcW, srcH);
  const dstBounds = getFaceBounds(dstLandmarks, width, height);

  log("bounds", { srcBounds, dstBounds });

  if (srcBounds.w < 10 || srcBounds.h < 10 || dstBounds.w < 10 || dstBounds.h < 10) {
    log("SKIP: bounds too small");
    return;
  }

  try {
    ctx.save();
    ctx.drawImage(
      srcImg,
      srcBounds.x, srcBounds.y, srcBounds.w, srcBounds.h,
      dstBounds.x, dstBounds.y, dstBounds.w, dstBounds.h
    );
    ctx.restore();

    ctx.save();
    ctx.font = "bold 16px sans-serif";
    ctx.fillStyle = "rgba(0,255,0,0.8)";
    ctx.fillText("SWAP", 8, 22);
    ctx.restore();

    if (typeof window !== "undefined" && window.DEBUG_FACESWAP) {
      ctx.save();
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 3;
      ctx.strokeRect(dstBounds.x, dstBounds.y, dstBounds.w, dstBounds.h);
      ctx.fillStyle = "red";
      ctx.fillRect(dstBounds.x + dstBounds.w / 2 - 5, dstBounds.y + dstBounds.h / 2 - 5, 10, 10);
      ctx.restore();
    }
    log("draw complete");
  } catch (err) {
    console.error("[FaceSwap:overlay] draw error", err);
    ctx.restore();
  }
}
