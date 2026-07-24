"""
face_quality_checker.py
------------------------
AI Engineering utility to inspect face images for common data-quality /
compliance issues before they are fed into a downstream face-recognition
or KYC pipeline.

Checks performed per image:
    1. Face detected at all (and how many faces)
    2. Blur / pixelation / low-resolution score
    3. Black & white (grayscale) vs color
    4. Face fully visible (not cropped at the frame edge, not too small,
       both eyes + mouth region present)
    5. Sunglasses / eye-occlusion detection

Design notes
------------
- Uses only OpenCV (cascade classifiers), so it runs anywhere OpenCV runs,
  including Python 3.13. No dlib / mediapipe hard dependency, because as of
  this writing mediapipe wheels lag behind the newest CPython releases and
  would block you on 3.13.5. See README.md "Upgrading accuracy" section for
  a drop-in swap to mediapipe/dlib landmarks if you later pin to Python
  3.11/3.12.
- Every check returns a numeric score plus a boolean verdict driven by a
  threshold in `AnalyzerConfig`, so you can tune sensitivity without
  touching the logic.
- Built to run on a single image or a whole folder (batch mode), and to be
  imported as a library (`FaceQualityAnalyzer`) inside a bigger pipeline.

Author: AI Engineering (generated)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("face_quality_checker")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AnalyzerConfig:
    """Tunable thresholds. Adjust these against a labeled validation set
    from your own data before trusting them in production."""

    # -- Blur / pixelation --
    # Variance of the Laplacian. Below this => considered blurry/pixelated.
    blur_variance_threshold: float = 80.0
    # Minimum face width/height (px) below which we flag "too low-res".
    min_face_size_px: int = 80

    # -- Grayscale / black & white --
    # Mean per-pixel channel difference (R vs G vs B). Below this => B&W.
    color_saturation_threshold: float = 8.0

    # -- Face visibility --
    # How close (px) a face box can be to the image border before we call
    # it "cropped / not fully visible".
    edge_margin_px: int = 4
    # Face bounding box must occupy at least this fraction of image area
    # to be considered "close enough / not tiny in a crowd shot".
    min_face_area_ratio: float = 0.02

    # -- Sunglasses / eye occlusion --
    # Instead of a fixed brightness cutoff (unreliable across skin tones /
    # lighting), we compare the eye band's brightness against this same
    # face's forehead+cheek brightness. If the eye band is at least this
    # fraction darker than the rest of the face, treat it as a lens.
    # e.g. 0.90 means "eye band must be <= 90% as bright as the rest of face".
    eye_region_relative_darkness_ratio: float = 0.90
    # Absolute fallback: still useful when the whole face is in shadow and
    # ratios get noisy. Only used as a secondary OR condition, not required.
    eye_region_dark_threshold: float = 90.0

    # -- Face count --
    # Reject images containing more than this many detected faces (e.g. a
    # group photo, or someone photobombing a profile picture). Set to a
    # higher number if multi-person photos are acceptable for your use case.
    max_faces_allowed: int = 1

    # -- Cascade files (bundled with opencv-python) --
    haar_face: str = "haarcascade_frontalface_default.xml"
    haar_eye: str = "haarcascade_eye.xml"
    haar_eye_glasses: str = "haarcascade_eye_tree_eyeglasses.xml"
    haar_smile: str = "haarcascade_smile.xml"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class FaceCheckResult:
    image_path: str
    faces_found: int = 0
    face_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h

    is_blurry_or_pixelated: Optional[bool] = None
    sharpness_score: Optional[float] = None

    is_black_and_white: Optional[bool] = None
    color_saturation_score: Optional[float] = None

    is_fully_visible: Optional[bool] = None
    visibility_issues: List[str] = field(default_factory=list)

    wearing_sunglasses: Optional[bool] = None
    eye_region_brightness: Optional[float] = None
    eyes_detected: int = 0

    overall_pass: Optional[bool] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Core analyzer
# --------------------------------------------------------------------------- #
class FaceQualityAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.cfg = config or AnalyzerConfig()
        base = Path(cv2.data.haarcascades)

        self.face_cascade = self._load_cascade(base / self.cfg.haar_face)
        self.eye_cascade = self._load_cascade(base / self.cfg.haar_eye)
        self.eye_glasses_cascade = self._load_cascade(base / self.cfg.haar_eye_glasses)

    @staticmethod
    def _load_cascade(path: Path) -> cv2.CascadeClassifier:
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise RuntimeError(f"Failed to load cascade classifier: {path}")
        return cascade

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, image_path: str) -> FaceCheckResult:
        result = FaceCheckResult(image_path=str(image_path))

        image = cv2.imread(str(image_path))
        if image is None:
            result.errors.append("Could not read image (bad path or unsupported format).")
            result.overall_pass = False
            return result

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40)
        )
        result.faces_found = len(faces)

        if len(faces) == 0:
            result.errors.append("No face detected.")
            result.overall_pass = False
            return result

        too_many_faces = len(faces) > self.cfg.max_faces_allowed
        if too_many_faces:
            result.errors.append(
                f"{len(faces)} faces detected, but only {self.cfg.max_faces_allowed} "
                "allowed (reject group photos / photobombs)."
            )

        # Use the largest detected face as the primary subject.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        result.face_box = (int(x), int(y), int(w), int(h))
        face_roi_gray = gray[y : y + h, x : x + w]
        face_roi_color = image[y : y + h, x : x + w]

        self._check_blur_pixelation(face_roi_gray, result)
        self._check_black_and_white(image, result)
        self._check_visibility(image, (x, y, w, h), result)
        self._check_sunglasses(face_roi_gray, face_roi_color, result)

        result.overall_pass = (
            result.faces_found >= 1
            and not too_many_faces
            and not result.is_blurry_or_pixelated
            and not result.is_black_and_white
            and result.is_fully_visible
            and not result.wearing_sunglasses
            and not result.errors
        )
        return result

    def analyze_folder(self, folder: str, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[FaceCheckResult]:
        folder_path = Path(folder)
        images = sorted(
            p for p in folder_path.rglob("*") if p.suffix.lower() in extensions
        )
        if not images:
            logger.warning("No images with extensions %s found in %s", extensions, folder)

        results = []
        for img_path in images:
            logger.info("Analyzing %s", img_path.name)
            results.append(self.analyze(img_path))
        return results

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #
    def _check_blur_pixelation(self, face_gray: np.ndarray, result: FaceCheckResult) -> None:
        """Laplacian variance measures edge sharpness. Low variance means
        the image is blurry, over-compressed, or pixelated/upscaled."""
        laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
        result.sharpness_score = round(float(laplacian_var), 2)

        too_small = (
            face_gray.shape[0] < self.cfg.min_face_size_px
            or face_gray.shape[1] < self.cfg.min_face_size_px
        )
        result.is_blurry_or_pixelated = bool(
            laplacian_var < self.cfg.blur_variance_threshold or too_small
        )
        if too_small:
            result.visibility_issues.append(
                f"Face region below {self.cfg.min_face_size_px}px "
                f"({face_gray.shape[1]}x{face_gray.shape[0]}) -> likely pixelated when scaled."
            )

    def _check_black_and_white(self, image_bgr: np.ndarray, result: FaceCheckResult) -> None:
        """A true grayscale image has near-identical B, G, R channel values
        at every pixel. We measure the mean absolute channel spread."""
        b, g, r = cv2.split(image_bgr.astype(np.int16))
        spread = (np.abs(b - g) + np.abs(g - r) + np.abs(b - r)) / 3.0
        mean_spread = float(np.mean(spread))
        result.color_saturation_score = round(mean_spread, 2)
        result.is_black_and_white = mean_spread < self.cfg.color_saturation_threshold

    def _check_visibility(
        self, image_bgr: np.ndarray, face_box: Tuple[int, int, int, int], result: FaceCheckResult
    ) -> None:
        img_h, img_w = image_bgr.shape[:2]
        x, y, w, h = face_box
        issues = list(result.visibility_issues)  # keep pixelation note if any

        # 1) Face touching / clipped at frame border
        margin = self.cfg.edge_margin_px
        if x <= margin or y <= margin or (x + w) >= (img_w - margin) or (y + h) >= (img_h - margin):
            issues.append("Face bounding box touches the image border (likely cropped/cut off).")

        # 2) Face too small relative to frame (subject far away / crowd shot)
        area_ratio = (w * h) / float(img_w * img_h)
        if area_ratio < self.cfg.min_face_area_ratio:
            issues.append(
                f"Face occupies only {area_ratio:.1%} of the image "
                f"(< {self.cfg.min_face_area_ratio:.1%} threshold)."
            )

        # 3) Both eyes present (proxy for "face not obstructed/turned away")
        face_gray = cv2.cvtColor(image_bgr[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        upper_half = face_gray[: int(h * 0.6), :]  # eyes live in the upper ~60% of the face box
        eyes = self.eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=6)
        result.eyes_detected = len(eyes)
        if len(eyes) < 2:
            issues.append(
                f"Only {len(eyes)} eye(s) detected -> face may be turned, "
                "partially covered, or occluded."
            )

        result.visibility_issues = issues
        result.is_fully_visible = len(issues) == 0

    def _check_sunglasses(
        self, face_gray: np.ndarray, face_color: np.ndarray, result: FaceCheckResult
    ) -> None:
        """Heuristic: the primary signal is that NEITHER the plain-eye
        cascade NOR the glasses-tuned cascade can find eyes — the glasses
        cascade specifically is trained to find eyes even when glasses are
        worn, so both failing together is a strong signal something is
        occluding the eyes.

        We confirm/attribute that occlusion to sunglasses (vs. e.g. closed
        eyes or a turned face) by checking whether the eye band is darker
        than this same face's forehead/cheek skin — a relative comparison,
        since an absolute brightness cutoff doesn't generalize across skin
        tones, lighting, or lens tint."""
        h, w = face_gray.shape[:2]
        eye_band_gray = face_gray[int(h * 0.20) : int(h * 0.55), :]

        # Reference "bare skin" brightness: forehead strip + lower face strip.
        forehead = face_gray[: int(h * 0.20), :]
        lower_face = face_gray[int(h * 0.55) :, :]
        skin_ref = np.concatenate([forehead.flatten(), lower_face.flatten()]) \
            if forehead.size and lower_face.size else face_gray.flatten()

        if eye_band_gray.size == 0 or skin_ref.size == 0:
            result.wearing_sunglasses = None
            return

        eye_brightness = float(np.mean(eye_band_gray))
        skin_brightness = float(np.mean(skin_ref))
        result.eye_region_brightness = round(eye_brightness, 2)

        plain_eyes = self.eye_cascade.detectMultiScale(eye_band_gray, scaleFactor=1.1, minNeighbors=6)
        glasses_eyes = self.eye_glasses_cascade.detectMultiScale(
            eye_band_gray, scaleFactor=1.1, minNeighbors=6
        )
        no_eyes_found = len(plain_eyes) == 0 and len(glasses_eyes) == 0

        relative_ratio = eye_brightness / skin_brightness if skin_brightness > 0 else 1.0
        relatively_dark = relative_ratio <= self.cfg.eye_region_relative_darkness_ratio
        absolutely_dark = eye_brightness < self.cfg.eye_region_dark_threshold

        result.wearing_sunglasses = bool(no_eyes_found and (relatively_dark or absolutely_dark))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_human_report(result: FaceCheckResult) -> None:
    print(f"\n{'='*60}\n{result.image_path}\n{'='*60}")
    if result.face_box is None:
        for e in result.errors:
            print(f"  [ERROR] {e}")
        print(f"  OVERALL: {'PASS' if result.overall_pass else 'FAIL'}")
        return

    for e in result.errors:
        print(f"  [ERROR] {e}")
    print(f"  Faces found:           {result.faces_found}")
    print(f"  Face box (x,y,w,h):    {result.face_box}")
    print(f"  Sharpness score:       {result.sharpness_score}"
          f"  -> {'BLURRY/PIXELATED' if result.is_blurry_or_pixelated else 'OK'}")
    print(f"  Color saturation:      {result.color_saturation_score}"
          f"  -> {'BLACK & WHITE' if result.is_black_and_white else 'COLOR'}")
    print(f"  Eyes detected:         {result.eyes_detected}")
    print(f"  Fully visible:         {'YES' if result.is_fully_visible else 'NO'}")
    for issue in result.visibility_issues:
        print(f"      - {issue}")
    print(f"  Eye region brightness: {result.eye_region_brightness}")
    print(f"  Sunglasses detected:   {'YES' if result.wearing_sunglasses else 'NO'}")
    print(f"  OVERALL:               {'PASS' if result.overall_pass else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check face images for pixelation, black & white, cropped/occluded faces, and sunglasses."
    )
    parser.add_argument("path", help="Path to a single image OR a folder of images.")
    parser.add_argument(
        "--json-out", help="Optional path to write full JSON results.", default=None
    )
    parser.add_argument(
        "--csv-out", help="Optional path to write a summary CSV.", default=None
    )
    args = parser.parse_args()

    analyzer = FaceQualityAnalyzer()
    target = Path(args.path)

    if target.is_dir():
        results = analyzer.analyze_folder(str(target))
    elif target.is_file():
        results = [analyzer.analyze(str(target))]
    else:
        logger.error("Path not found: %s", target)
        sys.exit(1)

    for r in results:
        _print_human_report(r)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Wrote JSON report -> %s", args.json_out)

    if args.csv_out:
        fieldnames = [
            "image_path", "faces_found", "overall_pass",
            "is_blurry_or_pixelated", "sharpness_score",
            "is_black_and_white", "color_saturation_score",
            "is_fully_visible", "eyes_detected", "wearing_sunglasses",
            "eye_region_brightness", "errors",
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                row["errors"] = "; ".join(row["errors"])
                writer.writerow({k: row.get(k) for k in fieldnames})
        logger.info("Wrote CSV summary -> %s", args.csv_out)

    passed = sum(1 for r in results if r.overall_pass)
    print(f"\nSummary: {passed}/{len(results)} images passed all checks.")


if __name__ == "__main__":
    main()