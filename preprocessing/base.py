"""Профиль предобработки — общий контракт (раздел 6.1 ТЗ)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ProfileResult:
    status: str                        # см. раздел 9 ТЗ (словарь статусов)
    output_path: str | None
    fallback_from: str | None
    metadata: dict = field(default_factory=dict)
    debug_artifacts: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    error: str | None = None


class PreprocessProfile(ABC):
    name: str
    version: str

    @abstractmethod
    def process(self, image_path: str, output_stem: str | None = None) -> ProfileResult:
        """output_stem — имя выходного файла без расширения.

        ОБЯЗАТЕЛЬНО передавать явно (slug для референсов, query_id для боевых
        фото) — у query-фото разных вин исходные имена совпадают (01.jpg,
        02.jpg...), поэтому имя выходного файла нельзя брать из имени
        исходника: кропы разных вин молча затрут друг друга.
        """
        ...
