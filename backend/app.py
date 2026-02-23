"""
FastAPI backend for face swap and AR processing.
"""
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from face_processor import FaceProcessor

processor: FaceProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    processor = FaceProcessor()
    yield
    if processor:
        processor.close()


app = FastAPI(
    title="Face Swap & AR Backend",
    description="Python backend for face swap, AR props, and landmarks",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_b64_image(b64: str):
    import numpy as np
    import cv2
    data = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_b64_image(img, fmt: str = "jpeg", quality: int = 85):
    import cv2
    if img is None:
        return ""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    _, buf = cv2.imencode(f".{fmt}", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": "python"}


@app.post("/api/debug")
async def debug_frame(frame: str = Form(...)):
    """Returns the frame with a debug overlay - use to verify pipeline."""
    frame_img = decode_b64_image(frame)
    if frame_img is None:
        raise HTTPException(status_code=400, detail="Invalid frame")
    import cv2
    out = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (10, 10), (200, 80), (0, 255, 0), 2)
    cv2.putText(out, "Backend OK", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return {"success": True, "image": encode_b64_image(out_rgb)}


@app.post("/api/process")
async def process_frame(
    frame: str = Form(..., description="Base64-encoded frame image"),
    mode: str = Form("landmarks", description="landmarks | face-swap | ar-props | face-swap-multi"),
    reference_image: str | None = Form(None, description="Base64 reference for face-swap"),
    prop: str = Form("glasses", description="AR prop: glasses, hat, crown, sunglasses, mustache, bow"),
):
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not ready")
    frame_img = decode_b64_image(frame)
    if frame_img is None:
        raise HTTPException(status_code=400, detail="Invalid frame image")
    try:
        if mode == "landmarks":
            out = processor.process_landmarks(frame_img)
        elif mode == "ar-props":
            out = processor.process_ar_props(frame_img, prop=prop)
        elif mode == "face-swap":
            if not reference_image:
                raise HTTPException(status_code=400, detail="reference_image required for face-swap")
            ref_img = decode_b64_image(reference_image)
            if ref_img is None:
                raise HTTPException(status_code=400, detail="Invalid reference image")
            out = processor.process_face_swap(frame_img, ref_img)
        elif mode == "face-swap-multi":
            out = processor.process_face_swap_multi(frame_img)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
        b64_out = encode_b64_image(out)
        return {"success": True, "image": b64_out}
    except Exception as e:
        return {"success": False, "error": str(e), "image": encode_b64_image(frame_img)}
