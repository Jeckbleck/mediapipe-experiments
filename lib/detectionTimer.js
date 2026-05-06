// Canvas has CSS `transform: scaleX(-1)` — canvas-left appears at screen-right.
// Draw background at x=0 (canvas-left = screen-right) and counter-flip the text
// so the CSS flip makes it readable.
export function drawTimer(ctx, ms, w, h) {
  const text = `${ms.toFixed(1)} ms`;
  const fontSize = Math.max(18, Math.round(h * 0.042));
  ctx.save();
  ctx.font = `bold ${fontSize}px monospace`;
  const pad = 10;
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(0, h - fontSize - pad * 2, tw + pad * 2, fontSize + pad * 2);
  ctx.translate(pad + tw, h - pad);
  ctx.scale(-1, 1);
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "#4ade80";
  ctx.fillText(text, 0, 0);
  ctx.restore();
}
