export function createPersonCutout(bgCategory = 0) {
  const maskCanvas = document.createElement("canvas");
  const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
  const smoothCanvas = document.createElement("canvas");
  const smoothCtx = smoothCanvas.getContext("2d");
  const personCanvas = document.createElement("canvas");
  const personCtx = personCanvas.getContext("2d");
  let maskImageData = null;

  return function cutout(video, mask, w, h, smooth = true) {
    const mw = mask.width;
    const mh = mask.height;
    const data = mask.getAsUint8Array();

    if (maskCanvas.width !== mw || maskCanvas.height !== mh) {
      maskCanvas.width = mw;
      maskCanvas.height = mh;
      maskImageData = null;
    }
    if (!maskImageData) maskImageData = maskCtx.createImageData(mw, mh);

    const px = maskImageData.data;
    for (let i = 0; i < data.length; i++) {
      const j = i * 4;
      px[j] = px[j + 1] = px[j + 2] = 255;
      px[j + 3] = data[i] !== bgCategory ? 255 : 0;
    }
    maskCtx.putImageData(maskImageData, 0, 0);

    const activeMask = smooth ? smoothCanvas : maskCanvas;
    if (smooth) {
      if (smoothCanvas.width !== mw || smoothCanvas.height !== mh) {
        smoothCanvas.width = mw;
        smoothCanvas.height = mh;
      }
      smoothCtx.clearRect(0, 0, mw, mh);
      smoothCtx.filter = "blur(2px)";
      smoothCtx.drawImage(maskCanvas, 0, 0);
      smoothCtx.filter = "none";
    }

    if (personCanvas.width !== w || personCanvas.height !== h) {
      personCanvas.width = w;
      personCanvas.height = h;
    }

    // Find the sub-region of activeMask that corresponds to the video aspect ratio
    const scale = Math.min(activeMask.width / w, activeMask.height / h);
    const sw = Math.round(w * scale);
    const sh = Math.round(h * scale);
    const ox = Math.round((activeMask.width - sw) / 2);
    const oy = Math.round((activeMask.height - sh) / 2);

    personCtx.clearRect(0, 0, w, h);
    personCtx.drawImage(video, 0, 0, w, h);
    personCtx.globalCompositeOperation = "destination-in";
    personCtx.drawImage(activeMask, ox, oy, sw, sh, 0, 0, w, h);
    personCtx.globalCompositeOperation = "source-over";

    return personCanvas;
  };
}
