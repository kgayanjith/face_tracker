# Face Quality Checker

A lightweight, dependency-light utility to screen face images for common
data-quality problems before they hit a face-recognition, KYC, or profile-photo
pipeline.

## Checks performed

| Check | How | Result field |
|---|---|---|
| Face present | Haar cascade frontal-face detector | `faces_found`, `face_box` |
| Blurry / pixelated / low-res | Laplacian variance (edge sharpness) + minimum face pixel size | `is_blurry_or_pixelated`, `sharpness_score` |
| Black & white | Mean channel spread across B/G/R (true grayscale has ~0 spread) | `is_black_and_white`, `color_saturation_score` |
| Face fully visible | Bounding box not touching frame edge, face not too small in frame, both eyes detected | `is_fully_visible`, `visibility_issues` |
| Sunglasses | Eye cascade (plain + glasses-tuned) fails to find eyes **and** the eye band is dark/flat (uniform lens, not skin/eyebrow texture) | `wearing_sunglasses`, `eye_region_brightness` |

Each image gets an `overall_pass` boolean (True only if it clears every check).

## Requirements

- Python 3.13.5 (also works on 3.9–3.13)
- `pip install -r requirements.txt`

Only OpenCV + NumPy are required. No dlib / mediapipe, so there's nothing
that can fail to build a wheel for a brand-new Python version.

## Usage

Single image:
```bash
python face_quality_checker.py path/to/photo.jpg
```

Folder (batch mode, recurses into subfolders):
```bash
python face_quality_checker.py path/to/folder --json-out results.json --csv-out results.csv
```

As a library, inside your own pipeline:
```python
from face_quality_checker import FaceQualityAnalyzer, AnalyzerConfig

analyzer = FaceQualityAnalyzer()
result = analyzer.analyze("photo.jpg")

if not result.overall_pass:
    print(result.errors, result.visibility_issues)
```

## Tuning thresholds

All thresholds live in `AnalyzerConfig` at the top of `face_quality_checker.py`:

```python
from face_quality_checker import FaceQualityAnalyzer, AnalyzerConfig

cfg = AnalyzerConfig(
    blur_variance_threshold=100.0,   # stricter blur cutoff
    min_face_size_px=120,            # require a bigger face
    eye_region_dark_threshold=70.0,  # sunglasses sensitivity
)
analyzer = FaceQualityAnalyzer(cfg)
```

Run the tool against a labeled sample of your own good/bad images first and
adjust these numbers — Haar-cascade-based heuristics are fast and dependency-free
but not as precise as a trained classifier, so calibration matters.

## Known limitations (be upfront about these with stakeholders)

- **Haar cascades** are lighter-weight but less accurate than modern DNN-based
  detectors, especially on non-frontal poses, unusual lighting, or diverse
  skin tones. For production-grade accuracy, swap `haar_face` detection for
  an OpenCV DNN face detector (e.g. `res10_300x300_ssd_iter_140000.caffemodel`)
  or a `mediapipe`/`dlib` landmark model — the `FaceQualityAnalyzer` class is
  structured so you only need to replace `_check_visibility` /
  `_check_sunglasses` internals, the public API stays the same.
- **Sunglasses detection is a heuristic**, not a trained classifier: it infers
  "sunglasses" from "no eyes found + dark, low-texture eye region." Very dark
  prescription glasses, heavy shadows, or closed eyes can produce false
  positives; reflective/mirrored sunglasses in bright light can occasionally
  produce false negatives. For higher accuracy, replace this step with a
  small CNN classifier trained on labeled sunglasses/no-sunglasses crops.
- **Pixelation vs. genuine blur** are not distinguished — both lower the
  Laplacian variance score. If you need to tell them apart specifically
  (e.g. to catch upscaled/interpolated images), add a block-artifact/FFT
  periodicity check on top of this.

## Suggested next steps for production hardening

1. Swap the face detector for an OpenCV DNN or `mediapipe` model once you can
   pin a Python version mediapipe fully supports (as of writing, mediapipe
   wheel availability lags the newest CPython release; check before pinning
   to 3.13.5 in production).
2. Train a small sunglasses/no-sunglasses classifier on your own labeled data
   instead of the brightness heuristic if false positive/negative rates
   matter for compliance decisions.
3. Add unit tests with a fixed set of known-good/known-bad sample images and
   wire this into CI so threshold changes are caught by regression tests.
4. Log `sharpness_score` / `color_saturation_score` distributions across your
   real dataset to pick thresholds statistically instead of by eye.
