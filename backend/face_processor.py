"""
Face processing using MediaPipe Tasks + OpenCV: face swap, AR props, landmarks.
Uses Face Landmarker (Tasks API) - MediaPipe 0.10.31+ removed Solutions API.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from scipy.spatial import Delaunay
from typing import Optional, List, Tuple
import base64
import io
import os

# Face Landmarker model - download from Google
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# MediaPipe Face Mesh triangulation indices (subset for face swap)
TRIANGULATION_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109, 33, 263, 61, 291, 0, 78, 13, 14, 17, 84, 181,
    91, 146, 77, 76, 62, 96, 89, 90, 43, 57, 202, 204, 106, 194, 182, 83, 201,
    18, 313, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
    146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 95, 88, 178, 87, 14, 317,
    402, 318, 324, 308, 263, 249, 390, 373, 374, 380, 381, 382, 362, 33, 7,
    163, 144, 145, 153, 154, 155, 133, 276, 283, 282, 295, 285, 46, 53, 52,
    65, 55, 70, 63, 105, 66, 107,
]

# Landmark indices for AR props
LEFT_EYE = [33, 133]
RIGHT_EYE = [263, 362]
NOSE_BRIDGE = 6
FOREHEAD = 10
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceProcessor:
    def __init__(self):
        model_path = _ensure_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=2,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def _decode_image(self, b64: str) -> np.ndarray:
        data = base64.b64decode(b64.split(",")[-1] if "," in b64 else b64)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def _encode_image(self, img: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> str:
        if img is None:
            return ""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        _, buf = cv2.imencode(f".{fmt}", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    def _get_landmarks(self, img: np.ndarray) -> list:
        h, w = img.shape[:2]
        img_cont = np.ascontiguousarray(img)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_cont)
        results = self.face_landmarker.detect(mp_image)
        if not results.face_landmarks:
            return []
        out = []
        for face_lm in results.face_landmarks:
            pts = [(float(l.x) * w, float(l.y) * h) for l in face_lm]
            out.append(pts)
        return out

    def _get_triangulation_points(self, landmarks: list, w: int, h: int) -> np.ndarray:
        points = []
        for i in TRIANGULATION_INDICES:
            if i < len(landmarks):
                x, y = landmarks[i]
                points.append([x, y])
        return np.array(points, dtype=np.float32) if points else np.array([])

    def _warp_triangle(self, src: np.ndarray, dst: np.ndarray, src_tri: np.ndarray, dst_tri: np.ndarray):
        r = cv2.boundingRect(np.float32([dst_tri]))
        x, y, w, h = r
        if w <= 0 or h <= 0:
            return
        dst_tri_shift = dst_tri - np.array([[x, y]], dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_shift), 255)
        warp = cv2.getAffineTransform(np.float32(dst_tri_shift), np.float32(src_tri))
        warped = cv2.warpAffine(
            src,
            warp,
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        roi = dst[y : y + h, x : x + w]
        roi[mask > 0] = warped[mask > 0]

    def _draw_landmarks_on_image(self, rgb_image: np.ndarray, detection_result) -> np.ndarray:
        """Draw face landmarks using official MediaPipe drawing_utils (per notebook)."""
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)
        for face_landmarks in face_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        return annotated_image

    def process_landmarks(self, img: np.ndarray) -> np.ndarray:
        """Draw face mesh landmarks and connections on image (RGB in/out). Uses official MediaPipe drawing."""
        img_cont = np.ascontiguousarray(img)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_cont)
        detection_result = self.face_landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            return img.copy()
        return self._draw_landmarks_on_image(img, detection_result)

    def _get_face_bounds(self, landmarks: list, w: int, h: int, pad: float = 0.2):
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cw = (max_x - min_x) * (1 + pad)
        ch = (max_y - min_y) * (1 + pad)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        x = max(0, int(cx - cw / 2))
        y = max(0, int(cy - ch / 2))
        sw = min(w - x, int(cw))
        sh = min(h - y, int(ch))
        return x, y, sw, sh

    def process_face_swap(
        self,
        frame: np.ndarray,
        reference_img: np.ndarray,
    ) -> np.ndarray:
        """Swap face from reference onto frame."""
        ref_landmarks_list = self._get_landmarks(reference_img)
        frame_landmarks_list = self._get_landmarks(frame)
        if not frame_landmarks_list or not ref_landmarks_list:
            return frame
        reference_landmarks = ref_landmarks_list[0]
        out = frame.copy()
        ref_h, ref_w = reference_img.shape[:2]
        f_h, f_w = frame.shape[:2]
        ref_pts = self._get_triangulation_points(reference_landmarks, ref_w, ref_h)
        frame_pts = self._get_triangulation_points(frame_landmarks_list[0], f_w, f_h)
        if len(ref_pts) != len(frame_pts) or len(ref_pts) < 4:
            rx, ry, rw, rh = self._get_face_bounds(reference_landmarks, ref_w, ref_h)
            fx, fy, fw, fh = self._get_face_bounds(frame_landmarks_list[0], f_w, f_h)
            if rw > 10 and rh > 10 and fw > 10 and fh > 10:
                ref_roi = reference_img[ry : ry + rh, rx : rx + rw]
                ref_scaled = cv2.resize(ref_roi, (fw, fh))
                out[fy : fy + fh, fx : fx + fw] = ref_scaled
            return out
        try:
            tri = Delaunay(frame_pts)
            for simplex in tri.simplices:
                src_tri = ref_pts[simplex]
                dst_tri = frame_pts[simplex]
                self._warp_triangle(reference_img, out, src_tri, dst_tri)
        except Exception:
            rx, ry, rw, rh = self._get_face_bounds(reference_landmarks, ref_w, ref_h)
            fx, fy, fw, fh = self._get_face_bounds(frame_landmarks_list[0], f_w, f_h)
            if rw > 10 and rh > 10 and fw > 10 and fh > 10:
                ref_roi = reference_img[ry : ry + rh, rx : rx + rw]
                ref_scaled = cv2.resize(ref_roi, (fw, fh))
                out[fy : fy + fh, fx : fx + fw] = ref_scaled
        return out

    def process_ar_props(self, img: np.ndarray, prop: str = "glasses") -> np.ndarray:
        """Draw AR prop (glasses, hat, crown, etc.) on face."""
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        landmarks_list = self._get_landmarks(img)
        if not landmarks_list:
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        lm = landmarks_list[0]
        h, w = img.shape[:2]
        if len(lm) < 365:
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        le = lm[33]
        ri = lm[362]
        nb = lm[6]
        ft = lm[10]
        eye_w = np.hypot(ri[0] - le[0], ri[1] - le[1]) * 1.4
        cy = (le[1] + ri[1]) / 2
        scale = max(2, eye_w / 40)
        if prop == "glasses":
            cv2.rectangle(out, (int(le[0] - eye_w * 0.6), int(cy - eye_w * 0.35)), (int(le[0]), int(cy + eye_w * 0.15)), (50, 50, 50), 2)
            cv2.rectangle(out, (int(ri[0]), int(cy - eye_w * 0.35)), (int(ri[0] + eye_w * 0.6), int(cy + eye_w * 0.15)), (50, 50, 50), 2)
            cv2.line(out, (int(le[0]), int(cy - eye_w * 0.1)), (int(ri[0]), int(cy - eye_w * 0.1)), (50, 50, 50), 2)
            cv2.circle(out, (int(nb[0]), int(nb[1])), 4, (30, 30, 30), -1)
        elif prop == "sunglasses":
            cv2.rectangle(out, (int(le[0] - eye_w * 0.65), int(cy - eye_w * 0.4)), (int(le[0] + eye_w * 0.05), int(cy + eye_w * 0.2)), (0, 0, 0), -1)
            cv2.rectangle(out, (int(ri[0] - eye_w * 0.05), int(cy - eye_w * 0.4)), (int(ri[0] + eye_w * 0.65), int(cy + eye_w * 0.2)), (0, 0, 0), -1)
            cv2.line(out, (int(le[0] + eye_w * 0.05), int(cy - eye_w * 0.1)), (int(ri[0] - eye_w * 0.05), int(cy - eye_w * 0.1)), (50, 50, 50), 2)
        elif prop == "hat":
            hat_y = ft[1] - eye_w * 0.8
            hat_w = eye_w * 2.2
            cx = (le[0] + ri[0]) / 2
            cv2.ellipse(out, (int(cx), int(hat_y)), (int(hat_w / 2), int(eye_w * 0.4)), 0, 0, 360, (139, 69, 19), -1)
            cv2.rectangle(out, (int(cx - hat_w / 2 - 5), int(hat_y - eye_w * 0.35)), (int(cx + hat_w / 2 + 5), int(hat_y)), (101, 67, 33), -1)
        elif prop == "crown":
            cx = (le[0] + ri[0]) / 2
            cy_c = ft[1] - eye_w * 0.9
            pts = np.array([[(cx - eye_w * 1.1, cy_c + eye_w * 0.3), (cx - eye_w * 0.5, cy_c), (cx, cy_c - eye_w * 0.5), (cx + eye_w * 0.5, cy_c), (cx + eye_w * 1.1, cy_c + eye_w * 0.3)]], dtype=np.int32)
            cv2.fillPoly(out, pts, (255, 215, 0))
            cv2.polylines(out, pts, True, (184, 134, 11), 2)
        elif prop == "mustache":
            lm_m = lm[61]
            rm_m = lm[291]
            ul = lm[13]
            my = (ul[1] + (lm_m[1] + rm_m[1]) / 2) / 2
            pts = np.array([[(int(lm_m[0]), int(my)), (int((lm_m[0] + rm_m[0]) / 2), int(my + eye_w * 0.2)), (int(rm_m[0]), int(my))]], dtype=np.int32)
            cv2.polylines(out, [pts], False, (50, 50, 50), max(3, int(scale)))
        elif prop == "bow":
            bx = (le[0] + ri[0]) / 2
            by = ft[1] - eye_w * 0.5
            cv2.ellipse(out, (int(bx - eye_w * 0.25), int(by)), (int(eye_w * 0.2), int(eye_w * 0.35)), 20, 0, 360, (233, 30, 99), -1)
            cv2.ellipse(out, (int(bx + eye_w * 0.25), int(by)), (int(eye_w * 0.2), int(eye_w * 0.35)), -20, 0, 360, (233, 30, 99), -1)
            cv2.circle(out, (int(bx), int(by)), int(eye_w * 0.08), (194, 24, 91), -1)
        else:
            cv2.rectangle(out, (int(le[0] - eye_w * 0.6), int(cy - eye_w * 0.35)), (int(le[0]), int(cy + eye_w * 0.15)), (50, 50, 50), 2)
            cv2.rectangle(out, (int(ri[0]), int(cy - eye_w * 0.35)), (int(ri[0] + eye_w * 0.6), int(cy + eye_w * 0.15)), (50, 50, 50), 2)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def process_face_swap_multi(self, img: np.ndarray) -> np.ndarray:
        """Swap faces between two people in frame."""
        landmarks_list = self._get_landmarks(img)
        if len(landmarks_list) < 2:
            return img
        out = img.copy()
        h, w = img.shape[:2]
        pts_a = self._get_triangulation_points(landmarks_list[0], w, h)
        pts_b = self._get_triangulation_points(landmarks_list[1], w, h)
        if len(pts_a) != len(pts_b) or len(pts_a) < 4:
            ax, ay, aw, ah = self._get_face_bounds(landmarks_list[0], w, h)
            bx, by, bw, bh = self._get_face_bounds(landmarks_list[1], w, h)
            if aw > 10 and ah > 10 and bw > 10 and bh > 10:
                roi_a = img[ay : ay + ah, ax : ax + aw].copy()
                roi_b = img[by : by + bh, bx : bx + bw].copy()
                out[ay : ay + ah, ax : ax + aw] = cv2.resize(roi_b, (aw, ah))
                out[by : by + bh, bx : bx + bw] = cv2.resize(roi_a, (bw, bh))
            return out
        try:
            tri_a = Delaunay(pts_a)
            for simplex in tri_a.simplices:
                src_tri = pts_b[simplex]
                dst_tri = pts_a[simplex]
                self._warp_triangle(img, out, src_tri, dst_tri)
            tri_b = Delaunay(pts_b)
            for simplex in tri_b.simplices:
                src_tri = pts_a[simplex]
                dst_tri = pts_b[simplex]
                self._warp_triangle(img, out, src_tri, dst_tri)
        except Exception:
            ax, ay, aw, ah = self._get_face_bounds(landmarks_list[0], w, h)
            bx, by, bw, bh = self._get_face_bounds(landmarks_list[1], w, h)
            if aw > 10 and ah > 10 and bw > 10 and bh > 10:
                roi_a = img[ay : ay + ah, ax : ax + aw].copy()
                roi_b = img[by : by + bh, bx : bx + bw].copy()
                out[ay : ay + ah, ax : ax + aw] = cv2.resize(roi_b, (aw, ah))
                out[by : by + bh, bx : bx + bw] = cv2.resize(roi_a, (bw, bh))
        return out

    def close(self):
        if hasattr(self.face_landmarker, "close"):
            self.face_landmarker.close()
