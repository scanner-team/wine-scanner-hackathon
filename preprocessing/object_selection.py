"""Выбор главного объекта — общий модуль для BOTTLE_CROP и LABEL_CROP.

centrality = exp( -(d_center / (sigma * half_diagonal))^2 )
score      = confidence^w_conf  x  area_term^w_area  x  centrality^w_centrality

`area_term` зависит от `area_mode`:
  - "reward"   (BOTTLE_CROP): area_term = area_ratio — «крупнее и по центру»
    верно для бутылки среди соседей на полке.
  - "penalize" (LABEL_CROP): area_term = 1 - area_ratio — для этикетки крупный
    бокс обычно значит захват фона, поэтому крупнее должно быть хуже.
    `1 - area_ratio` (ограничен [0,1]) выбран вместо `1/area_ratio`, чтобы
    микроскопический ложный бокс не давал взрыва score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Candidate:
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 в пикселях исходного кадра
    confidence: float
    label: str | None = None  # текстовая метка кандидата (для label_crop)


def select_main_object(
    candidates: list[Candidate],
    image_size: tuple[int, int],
    w_confidence: float = 1.0,
    w_area: float = 1.0,
    w_centrality: float = 1.0,
    centrality_sigma: float = 0.35,
    area_mode: str = "reward",
) -> list[dict]:
    """Возвращает список кандидатов с посчитанным score, один помечен selected=True."""
    if area_mode not in ("reward", "penalize"):
        raise ValueError(f"Неизвестный area_mode: {area_mode!r} (ожидается 'reward' или 'penalize')")

    img_w, img_h = image_size
    img_area = img_w * img_h
    cx0, cy0 = img_w / 2.0, img_h / 2.0
    half_diag = math.hypot(img_w, img_h) / 2.0

    scored = []
    for c in candidates:
        x1, y1, x2, y2 = c.bbox
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / img_area if img_area else 0.0
        bcx, bcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d_center = math.hypot(bcx - cx0, bcy - cy0)
        centrality = math.exp(-((d_center / (centrality_sigma * half_diag)) ** 2)) if half_diag else 0.0
        conf = max(c.confidence, 1e-6)
        area_term = (1.0 - area_ratio) if area_mode == "penalize" else area_ratio
        area_term = max(area_term, 1e-6)
        cen = max(centrality, 1e-6)
        score = (conf ** w_confidence) * (area_term ** w_area) * (cen ** w_centrality)
        scored.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": c.confidence,
            "area_ratio": area_ratio,
            "area_mode": area_mode,
            "centrality": centrality,
            "selection_score": score,
            "label": c.label,
            "selected": False,
        })

    if scored:
        best = max(scored, key=lambda d: d["selection_score"])
        best["selected"] = True

    return scored
