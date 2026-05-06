import { FilesetResolver } from "@mediapipe/tasks-vision";

export const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";

let _promise = null;
export function getFileset() {
  if (!_promise) _promise = FilesetResolver.forVisionTasks(WASM_PATH);
  return _promise;
}
