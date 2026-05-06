// Canvas has CSS `transform: scaleX(-1)` — canvas-left appears at screen-right.
// Draw background at x=0 (canvas-left = screen-right) and counter-flip text
// so the CSS flip makes it readable.
export function createPerfMonitor() {
  let smoothMs = 0;
  let smoothFps = 0;
  let lastFrameTime = performance.now();

  return {
    draw(ctx, detectionMs, w, h) {
      const now = performance.now();
      const delta = now - lastFrameTime;
      lastFrameTime = now;

      smoothMs = smoothMs * 0.9 + detectionMs * 0.1;
      if (delta > 0 && delta < 500) {
        smoothFps = smoothFps * 0.9 + (1000 / delta) * 0.1;
      }

      const fontSize = Math.max(18, Math.round(h * 0.042));
      const gap = 4;
      const pad = 10;
      const lineH = fontSize + gap;

      ctx.save();
      ctx.font = `bold ${fontSize}px monospace`;

      const msText  = `${smoothMs.toFixed(1)} ms`;
      const fpsText = `${Math.round(smoothFps)} fps`;
      const innerW  = Math.max(ctx.measureText(msText).width, ctx.measureText(fpsText).width);
      const boxW    = innerW + pad * 2;
      const boxH    = lineH * 2 + pad * 2;

      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(0, h - boxH, boxW, boxH);

      ctx.textAlign    = "left";
      ctx.textBaseline = "bottom";

      // fps row (top) — color-coded: green ≥25, yellow ≥15, red <15
      const fps = Math.round(smoothFps);
      ctx.fillStyle = fps >= 25 ? "#4ade80" : fps >= 15 ? "#facc15" : "#f87171";
      ctx.save();
      ctx.translate(pad + ctx.measureText(fpsText).width, h - pad - lineH);
      ctx.scale(-1, 1);
      ctx.fillText(fpsText, 0, 0);
      ctx.restore();

      // ms row (bottom)
      ctx.fillStyle = "#4ade80";
      ctx.save();
      ctx.translate(pad + ctx.measureText(msText).width, h - pad);
      ctx.scale(-1, 1);
      ctx.fillText(msText, 0, 0);
      ctx.restore();

      ctx.restore();
    },
  };
}
