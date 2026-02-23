/**
 * Simple logger for face swap debugging
 * Set DEBUG_FACESWAP=true in console to enable verbose logs
 */
const DEBUG = () => typeof window !== "undefined" && window.DEBUG_FACESWAP === true;

export const log = {
  info: (...args) => console.log("[FaceSwap]", ...args),
  warn: (...args) => console.warn("[FaceSwap]", ...args),
  error: (...args) => console.error("[FaceSwap]", ...args),
  debug: (...args) => DEBUG() && console.log("[FaceSwap:debug]", ...args),
};
