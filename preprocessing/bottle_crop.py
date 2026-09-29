"""BOTTLE_CROP — YOLO11, класс COCO `bottle` (раздел 7.2 ТЗ)."""
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageOps

from preprocessing.base import PreprocessProfile, ProfileResult
from preprocessing.object_selection import Candidate, select_main_object

_model_cache: dict[str, object] = {}


def _get_model(weights: str):
    if weights not in _model_cache:
        from ultralytics import YOLO
        _model_cache[weights] = YOLO(weights)
    return _model_cache[weights]


class BottleCropProfile(PreprocessProfile):
    name = "bottle_crop"
    version = "1"

    def __init__(
        self,
        output_dir: Path,
        weights: str = "yolo11l.pt",
        confidence_threshold: float = 0.25,
        low_confidence_threshold: float = 0.40,
        padding: float = 0.08,
        selection_weights: dict | None = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.padding = padding
        self.selection_weights = selection_weights or {}

    def process(self, image_path: str, output_stem: str | None = None) -> ProfileResult:
        t0 = time.time()
        src = Path(image_path)
        stem = output_stem or src.stem
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                model = _get_model(self.weights)
                results = model.predict(im, classes=[39], conf=self.confidence_threshold, verbose=False)
                r = results[0]
                candidates = []
                for box in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    candidates.append(Candidate(bbox=(x1, y1, x2, y2), confidence=conf, label="bottle"))

                if not candidates:
                    return ProfileResult(
                        status="detection_failed", output_path=None, fallback_from="raw",
                        metadata={"model": self.weights, "class": "bottle", "num_detections": 0,
                                  "source_size_px": [w, h]},
                        processing_time_ms=(time.time() - t0) * 1000,
                    )

                scored = select_main_object(candidates, (w, h), **self.selection_weights)
                best = next(c for c in scored if c["selected"])
                x1, y1, x2, y2 = best["bbox"]
                bw, bh = x2 - x1, y2 - y1
                x1 -= bw * self.padding
                x2 += bw * self.padding
                y1 -= bh * self.padding
                y2 += bh * self.padding
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = im.crop((int(x1), int(y1), int(x2), int(y2)))
                out_path = self.output_dir / f"{stem}.jpg"
                crop.save(out_path, "JPEG", quality=95)

                status = "ok"
                if best["confidence"] < self.low_confidence_threshold:
                    status = "low_confidence"

                return ProfileResult(
                    status=status,
                    output_path=str(out_path),
                    fallback_from=None,
                    metadata={
                        "model": self.weights, "class": "bottle", "num_detections": len(candidates),
                        "all_candidates": scored, "selected_bbox": best["bbox"],
                        "confidence": best["confidence"], "padding": self.padding,
                        "crop_size_px": list(crop.size), "source_size_px": [w, h],
                    },
                    processing_time_ms=(time.time() - t0) * 1000,
                )
        except Exception as e:
            return ProfileResult(
                status="processing_error", output_path=None, fallback_from="raw",
                error=str(e), processing_time_ms=(time.time() - t0) * 1000,
            )
