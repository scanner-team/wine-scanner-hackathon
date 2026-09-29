"""Пиксельный бюджет — нормализация размера фото перед эмбеддингом.

Разрешение фото при кодировании в эмбеддинг должно совпадать для
референсов каталога и для запроса (иначе вектора несовместимы). Модуль
приводит кадр к целевому бюджету.

Режимы:
  - "area"      (по умолчанию): вписать в [min_pixels, max_pixels] по
    общей площади кадра.
  - "long_side": ресайз до заданной длинной стороны в пикселях, не по
    площади (для разных соотношений сторон даёт разную площадь).
  - "none": без ресайза, только RGB-конвертация.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def normalize_for_budget(
    image_path: str | Path,
    min_pixels: int,
    max_pixels: int,
    warn_below_px: int = 200,
    mode: str = "area",
    target_long_side: int | None = None,
) -> tuple[Image.Image, dict]:
    if mode not in ("area", "long_side", "none"):
        raise ValueError(f"Неизвестный mode пиксельного бюджета: {mode!r} (ожидается 'area', 'long_side' или 'none')")
    if mode == "long_side" and not target_long_side:
        raise ValueError("mode='long_side' требует target_long_side")

    with Image.open(image_path) as src:
        im = src.convert("RGB").copy()
    w, h = im.size
    meta = {"original_size_px": [w, h], "mode": mode}

    if mode == "long_side":
        long_side = max(w, h)
        if long_side != target_long_side:
            scale = target_long_side / long_side
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
            meta["upscaled"] = scale > 1
            meta["downscaled"] = scale < 1
        pixels = im.size[0] * im.size[1]
        # ресайз по длинной стороне не гарантирует попадание в
        # [min_pixels, max_pixels] — дожимаем при необходимости
        if pixels < min_pixels:
            scale2 = (min_pixels / pixels) ** 0.5
            new_size2 = (max(1, round(im.size[0] * scale2)), max(1, round(im.size[1] * scale2)))
            im = im.resize(new_size2, Image.LANCZOS)
            meta["upscaled"] = True
        elif pixels > max_pixels:
            scale2 = (max_pixels / pixels) ** 0.5
            new_size2 = (max(1, round(im.size[0] * scale2)), max(1, round(im.size[1] * scale2)))
            im = im.resize(new_size2, Image.LANCZOS)
            meta["downscaled"] = True
    elif mode == "area":
        pixels = w * h
        if pixels < min_pixels and pixels > 0:
            scale = (min_pixels / pixels) ** 0.5
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
            meta["upscaled"] = True
        elif pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
            meta["downscaled"] = True
    # mode == "none": никакого ресайза, картинка уходит как есть (см. докстринг)

    meta["final_size_px"] = list(im.size)
    if min(im.size) < warn_below_px:
        meta["low_resolution"] = True

    return im, meta
