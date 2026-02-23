import Delaunator from "delaunator";

// Subset of face landmark indices for triangulation (face oval + key features)
const TRIANGULATION_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109, 33, 263, 61, 291, 0, 78, 13, 14, 17, 84, 181,
  91, 146, 77, 76, 62, 96, 89, 90, 43, 57, 202, 204, 106, 194, 182, 83, 201,
  18, 313, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
  146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317,
  402, 318, 324, 308, 263, 249, 390, 373, 374, 380, 381, 382, 362, 33, 7,
  163, 144, 145, 153, 154, 155, 133, 276, 283, 282, 295, 285, 46, 53, 52,
  65, 55, 70, 63, 105, 66, 107,
];

/**
 * Convert normalized landmarks (0-1) to pixel coordinates
 */
export function toPixelCoords(landmarks, width, height) {
  return landmarks.map((lm) => ({
    x: lm.x * width,
    y: lm.y * height,
  }));
}

/**
 * Get a subset of landmarks for triangulation
 */
export function getTriangulationPoints(landmarks, width, height) {
  const points = [];
  const seen = new Set();
  for (const i of TRIANGULATION_INDICES) {
    if (i < landmarks.length && !seen.has(i)) {
      seen.add(i);
      const lm = landmarks[i];
      points.push({ x: lm.x * width, y: lm.y * height, i });
    }
  }
  return points;
}

/**
 * Create Delaunay triangulation from points
 */
export function triangulate(points) {
  const coords = points.flatMap((p) => [p.x, p.y]);
  const d = new Delaunator(coords);
  const triangles = [];
  for (let i = 0; i < d.triangles.length; i += 3) {
    triangles.push([d.triangles[i], d.triangles[i + 1], d.triangles[i + 2]]);
  }
  return { points, triangles };
}

/**
 * Barycentric coordinates: is point p inside triangle (a,b,c)?
 * Returns (u, v, w) or null if outside
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
 * Warp a triangle from source to destination
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
        const srcX = bc.u * srcTri[0].x + bc.v * srcTri[1].x + bc.w * srcTri[2].x;
        const srcY = bc.u * srcTri[0].y + bc.v * srcTri[1].y + bc.w * srcTri[2].y;
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
 * Swap faces between two sets of landmarks
 * srcImg: source image (video frame or image)
 * landmarksA, landmarksB: normalized landmarks for face A and B
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

  const pointsA = getTriangulationPoints(landmarksA, width, height);
  const pointsB = getTriangulationPoints(landmarksB, width, height);

  if (pointsA.length < 3 || pointsB.length < 3) return;

  const { points: ptsA, triangles } = triangulate(pointsA);
  const { points: ptsB } = triangulate(pointsB);

  if (ptsB.length !== ptsA.length) return;

  for (const tri of triangles) {
    const i0 = tri[0], i1 = tri[1], i2 = tri[2];
    const srcTri = [ptsB[i0], ptsB[i1], ptsB[i2]];
    const dstTri = [ptsA[i0], ptsA[i1], ptsA[i2]];
    warpTriangle(srcData, dstData, srcTri, dstTri, width, height, width, height);
  }

  dstCtx.putImageData(dstData, 0, 0);

  for (const tri of triangles) {
    const i0 = tri[0], i1 = tri[1], i2 = tri[2];
    const srcTri = [ptsA[i0], ptsA[i1], ptsA[i2]];
    const dstTri = [ptsB[i0], ptsB[i1], ptsB[i2]];
    warpTriangle(srcData, dstData, srcTri, dstTri, width, height, width, height);
  }

  dstCtx.putImageData(dstData, 0, 0);
  ctx.drawImage(dstCanvas, 0, 0);
}

/**
 * Get face bounding box from landmarks (with padding)
 */
function getFaceBounds(landmarks, width, height, padding = 0.25) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
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
