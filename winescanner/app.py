"""HTTP-сервис распознавания вина по фото — FastAPI + CUDA (embedding на
GPU через sentence-transformers, чтение этикетки — VLM через локальный
vLLM OpenAI-совместимый сервер). Подробности архитектуры — ARCHITECTURE.md
и EXPERIMENTS.md в корне репозитория; переменные окружения — .env.example
и docs/DEPLOYMENT.md.

BOTTLE_CROP (YOLO) — код присутствует как экспериментальная функция,
намеренно отключена по умолчанию флагом ENABLE_BOTTLE_CROP=False (не
проверялась в проде, включать только после отдельного теста на своих
данных; веса `yolo11l.pt` не входят в репозиторий — см. preprocessing/).

Запуск (см. docs/DEPLOYMENT.md за подробностями и переменными окружения):
    uvicorn winescanner.app:app --host 0.0.0.0 --port 8080
(предварительно должен быть поднят vLLM-сервер на порту из
WINESCANNER_VLLM_URL, по умолчанию localhost:8000)
"""
from __future__ import annotations

import base64
import csv
import difflib
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from retrievers.pixel_budget import normalize_for_budget

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val.strip().lower() in ("1", "true", "yes")


# BASE — корень репозитория (папка, где лежит requirements.txt), не сама
# папка winescanner/. По умолчанию вычисляется относительно расположения
# этого файла, чтобы работать из любой директории без ручной настройки;
# переопределяется WINESCANNER_BASE_DIR, если данные лежат отдельно от кода.
BASE = Path(os.environ.get("WINESCANNER_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(BASE))
from prompt_v4 import PROMPT_V4  # noqa: E402
CATALOG_NPZ = BASE / "data" / "catalog" / "vectors" / "references.npz"
CATALOG_MANIFEST = BASE / "data" / "catalog" / "vectors" / "references_full.csv"
CATALOG_VLM_RAW_DIR = BASE / "data" / "catalog" / "vlm_raw"
WINES_CSV = BASE / "data" / "catalog" / "wines.csv"
LOW_RES_CSV = BASE / "winescanner" / "catalog_low_res_photos.csv"
PHOTOS_DIR = BASE / "data" / "catalog" / "photos"
PHOTOS_THUMB_DIR = BASE / "data" / "catalog" / "photos_thumb"
STATIC_DIR = Path(__file__).parent / "static"
PRODUCER_ALIASES_JSON = BASE / "winescanner" / "producer_aliases.json"
GRAPE_ALIASES_JSON = BASE / "winescanner" / "grape_aliases.json"
TOKEN_DF_PKL = BASE / "winescanner" / "catalog_token_df.pkl"

QUERY_LOG_DIR = BASE / "winescanner" / "query_log"
BATCH_STAGING_DIR = BASE / "winescanner" / "batch_staging"
QUERY_CROP_DIR = BASE / "winescanner" / "query_crops"
YOLO_WEIGHTS = BASE / "yolo11l.pt"
PREDICT_LOG_PATH = BASE / "winescanner" / "predict_log.jsonl"
MARKS_LOG_PATH = BASE / "winescanner" / "marks_log.jsonl"
SCANNER_ERRORS_MD = BASE / "winescanner" / "SCANNER_ERRORS.md"
CONFIDENT_MISSES_MD = BASE / "winescanner" / "CONFIDENT_MISSES.md"
METRICS_MD_PATH = BASE / "docs" / "METRICS.md"

EMBED_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
VLM_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
VLLM_SERVER_URL = os.environ.get("WINESCANNER_VLLM_URL", "http://localhost:8000/v1/chat/completions")
MIN_PIXELS = 50_000
MAX_PIXELS = 2_000_000
CONFIDENCE_THRESHOLD = float(os.environ.get("WINESCANNER_CONFIDENCE_THRESHOLD", "0.76"))
TOP_K = int(os.environ.get("WINESCANNER_TOP_K", "5"))

# BOTTLE_CROP — экспериментальная функция, см. docstring файла. Выключена
# по умолчанию; веса yolo11l.pt не входят в репозиторий.
ENABLE_BOTTLE_CROP = _env_bool("WINESCANNER_ENABLE_BOTTLE_CROP", False)

# Ключ для /v1/predict (контрактный эндпоинт работает без ключа) —
# обязательно задать свой через .env, значение по умолчанию нерабочее.
# См. .env.example.
PERSONAL_API_KEY = os.environ.get("WINESCANNER_API_KEY", "changeme-generate-your-own-key")
ENABLE_CONTRACT_ENDPOINT = _env_bool("WINESCANNER_ENABLE_CONTRACT_ENDPOINT", False)

W_PRODUCER, W_GRAPE, W_CATEGORY, W_NAME, W_RAWTEXT = 3.0, 2.0, 1.0, 1.0, 3.0
TEXT_MAX = W_PRODUCER + W_GRAPE + W_CATEGORY + W_NAME + W_RAWTEXT
BETA = 0.15
MIN_TOKENS_FOR_FULL_TRUST = 6

QUERY_LOG_DIR.mkdir(parents=True, exist_ok=True)
BATCH_STAGING_DIR.mkdir(parents=True, exist_ok=True)
QUERY_CROP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Wine Scanner — GPU service")
_state: dict = {}
_batches: dict[str, dict] = {}
app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")
app.mount("/photos_thumb", StaticFiles(directory=str(PHOTOS_THUMB_DIR)), name="photos_thumb")
app.mount("/query_log", StaticFiles(directory=str(QUERY_LOG_DIR)), name="query_log")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/batch_staging", StaticFiles(directory=str(BATCH_STAGING_DIR)), name="batch_staging")

FIELD_RE = {
    k: re.compile(r'"' + k + r'"\s*:\s*(null|"(?:[^"\\]|\\.)*")')
    for k in ["visible_name", "producer", "cuvee", "vintage", "grape", "category"]
}


def lenient_parse(text: str) -> tuple[dict, bool]:
    try:
        return json.loads(text), True
    except Exception:
        pass
    out: dict = {}
    for key, rx in FIELD_RE.items():
        m = rx.search(text)
        if m:
            raw = m.group(1)
            out[key] = None if raw == "null" else json.loads(raw)
    return out, False


def norm_key(s) -> str:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    return re.sub(r"[^a-zа-яё0-9]", "", str(s or "").lower())


def _as_text(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v or "")


_COLOR_KEYWORDS = {
    "Белое": ["бел", "white", "blanc", "bianco"],
    "Красное": ["красн", "red", "rouge", "rosso"],
    "Розовое": ["розов", "rose", "rosé", "ros"],
    "Оранжевое": ["оранж", "orange", "янтарн", "amber"],
}


def extract_color(text) -> str | None:
    text = _as_text(text)
    if not text:
        return None
    low = text.lower()
    best_pos, best_color = None, None
    for color, keywords in _COLOR_KEYWORDS.items():
        for kw in keywords:
            pos = low.find(kw)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos, best_color = pos, color
    return best_color


SWEETNESS_TERMS = [
    ("экстра брют", "ekstra-bryut", 0), ("экстра-брют", "ekstra_bryut", 0),
    ("полусухое", "polusuhoe", 3),
    ("полусладкое", "polusladkoe", 4),
    ("брют", "bryut", 1),
    ("сухое", "suhoe", 2),
    ("сладкое", "sladkoe", 5),
]

GRAPE_PROFILES = {
    "каберне совиньон": {"acidity": 3, "tannin": 5, "body": 5},
    "шардоне": {"acidity": 3, "tannin": 0, "body": 4},
    "пино нуар": {"acidity": 4, "tannin": 2, "body": 2},
    "рислинг": {"acidity": 5, "tannin": 0, "body": 1},
    "рислинг рейнский": {"acidity": 5, "tannin": 0, "body": 1},
    "совиньон блан": {"acidity": 5, "tannin": 0, "body": 2},
    "саперави": {"acidity": 4, "tannin": 4, "body": 4},
    "мерло": {"acidity": 2, "tannin": 3, "body": 4},
    "каберне фран": {"acidity": 4, "tannin": 4, "body": 3},
    "ркацители": {"acidity": 4, "tannin": 0, "body": 2},
    "сира": {"acidity": 3, "tannin": 4, "body": 5},
    "шираз": {"acidity": 3, "tannin": 4, "body": 5},
    "алиготе": {"acidity": 4, "tannin": 0, "body": 1},
    "мускат": {"acidity": 2, "tannin": 0, "body": 1},
    "мускат белый": {"acidity": 2, "tannin": 0, "body": 1},
    "мускат оттонель": {"acidity": 2, "tannin": 0, "body": 1},
    "пино гри": {"acidity": 3, "tannin": 0, "body": 3},
    "пино гриджио": {"acidity": 3, "tannin": 0, "body": 2},
    "пино блан": {"acidity": 3, "tannin": 0, "body": 2},
    "красностоп золотовский": {"acidity": 3, "tannin": 4, "body": 4},
    "красностоп": {"acidity": 3, "tannin": 4, "body": 4},
    "кокур": {"acidity": 3, "tannin": 0, "body": 2},
    "кокур белый": {"acidity": 3, "tannin": 0, "body": 2},
    "мальбек": {"acidity": 3, "tannin": 4, "body": 5},
    "марселан": {"acidity": 3, "tannin": 4, "body": 4},
    "санджовезе": {"acidity": 4, "tannin": 4, "body": 3},
    "темпранильо": {"acidity": 3, "tannin": 4, "body": 4},
    "пти вердо": {"acidity": 3, "tannin": 5, "body": 5},
    "гевюрцтраминер": {"acidity": 2, "tannin": 0, "body": 3},
    "семильон": {"acidity": 3, "tannin": 0, "body": 3},
    "шенен блан": {"acidity": 4, "tannin": 0, "body": 3},
    "вионье": {"acidity": 2, "tannin": 0, "body": 4},
}

AROMA_FAMILIES = {
    "fruit": ["вишн", "слив", "яблок", "груш", "ягод", "персик", "малин", "ежевик",
              "черешн", "чернослив", "крыжовник", "клубник", "абрикос", "земляник",
              "шелковиц", "айв", "смородин", "джем"],
    "floral": ["цвет", "роз", "акаци", "луговых", "полевых", "липы", "жасмин"],
    "citrus": ["цитрус", "грейпфрут", "лайм", "цедр", "лимон", "апельсин",
               "тропич", "манго", "ананас"],
    "spice": ["пряност", "пряны", "специ", "перц", "корица", "гвоздик"],
}


def _sweetness_score(name: str, slug: str, description: str) -> int | None:
    cyr_text = f"{name} {description}".lower()
    lat_text = slug.lower()
    for cyr_term, lat_term, score in SWEETNESS_TERMS:
        if cyr_term in cyr_text or lat_term in lat_text:
            return score
    return None


def _grape_structure(grape: str) -> dict:
    parts = re.split(r"[,/&+]|(?:\s+(?:и|and)\s+)", grape, flags=re.IGNORECASE)
    normalized = [p.strip().lower() for p in parts]
    found = [GRAPE_PROFILES[p] for p in normalized if p in GRAPE_PROFILES]
    if not found:
        return {"acidity": None, "tannin": None, "body": None}
    return {
        k: round(sum(f[k] for f in found) / len(found), 1)
        for k in ("acidity", "tannin", "body")
    }


def _aroma_scores(description: str) -> dict:
    low = description.lower()
    out = {}
    for fam, keywords in AROMA_FAMILIES.items():
        matched = sum(1 for kw in keywords if kw in low)
        out[fam] = 0 if matched == 0 else (3 if matched == 1 else 5)
    return out


def _compute_taste_profile(name: str, slug: str, description: str, grape: str) -> dict:
    profile = {"sweetness": _sweetness_score(name, slug, description or "")}
    profile.update(_grape_structure(grape or ""))
    profile.update(_aroma_scores(description or ""))
    return profile


def _fuzzy_lookup(key: str, table: dict[str, str]) -> str | None:
    if not key:
        return None
    if key in table:
        return table[key]
    best_ratio, best_val = 0.0, None
    for k, v in table.items():
        ratio = difflib.SequenceMatcher(None, key, k).ratio()
        if ratio > best_ratio:
            best_ratio, best_val = ratio, v
    return best_val if best_ratio >= 0.82 else None


def normalize_producer(raw) -> str | None:
    return _fuzzy_lookup(norm_key(_as_text(raw)), _state["producer_aliases"])


def normalize_grapes_blend(raw) -> list[str]:
    raw = _as_text(raw)
    if not raw:
        return []
    parts = re.split(r"[•,/&+]|(?:\s+(?:и|and)\s+)", raw, flags=re.IGNORECASE)
    out: list[str] = []
    for p in parts:
        g = _fuzzy_lookup(norm_key(p.strip()), _state["grape_aliases"])
        if g and g not in out:
            out.append(g)
    return out


def idf(token: str) -> float:
    df = _state["token_df"].get(token, 0)
    n_docs = _state["token_n_docs"]
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def tokens_of(raw_list) -> set:
    if not raw_list:
        return set()
    out = set()
    for item in raw_list:
        for w in re.split(r"\s+", str(item)):
            k = norm_key(w)
            if len(k) >= 4:
                out.add(k)
    return out


def idf_overlap_score(query_tokens: set, cand_tokens: set) -> float:
    if not query_tokens:
        return 0.0
    shared = query_tokens & cand_tokens
    den = sum(idf(t) for t in query_tokens)
    return (sum(idf(t) for t in shared) / den) if den > 0 else 0.0


def load_candidate_fields(slug: str) -> dict:
    path = CATALOG_VLM_RAW_DIR / f"{slug}.json"
    if not path.exists():
        return {"producer": None, "grapes": set(), "color": None, "name_norm": "", "raw_tokens": set()}
    d = json.loads(path.read_text(encoding="utf-8"))
    p = d.get("parsed", {})
    return {
        "producer": normalize_producer(p.get("producer")),
        "grapes": set(normalize_grapes_blend(p.get("grape"))),
        "color": extract_color(p.get("category")),
        "name_norm": norm_key(_as_text(p.get("visible_name"))),
        "raw_tokens": tokens_of(p.get("raw_visible_text")),
    }


def text_score(query: dict, cand: dict) -> float:
    s = 0.0
    if query["producer"] and cand["producer"] and norm_key(query["producer"]) == norm_key(cand["producer"]):
        s += W_PRODUCER
    if query["grapes"] and cand["grapes"] and (query["grapes"] & cand["grapes"]):
        s += W_GRAPE
    if query["color"] and cand["color"] and query["color"] == cand["color"]:
        s += W_CATEGORY
    if query["name_norm"] and cand["name_norm"]:
        s += W_NAME * difflib.SequenceMatcher(None, query["name_norm"], cand["name_norm"]).ratio()
    confidence = min(1.0, len(cand["raw_tokens"]) / MIN_TOKENS_FOR_FULL_TRUST)
    s += W_RAWTEXT * confidence * idf_overlap_score(query["raw_tokens"], cand["raw_tokens"])
    return s


def _vlm_generate(image_path: str) -> str:
    """VLM-запрос через локальный vLLM OpenAI-совместимый сервер
    (VLLM_SERVER_URL, Qwen/Qwen3-VL-4B-Instruct, bf16) - выбран по
    прямому бенчмарку против bf16/bitsandbytes/FlashAttention-2, см.
    EXPERIMENTS.md."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    payload = {
        "model": VLM_MODEL_ID,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT_V4},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "max_tokens": 500,
        "temperature": 0.0,
    }
    resp = requests.post(VLLM_SERVER_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@app.on_event("startup")
def load_everything() -> None:
    t0 = time.time()
    print("Загрузка модели эмбеддинга...", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL_ID, device="cuda", trust_remote_code=True)

    bottle_crop = None
    if ENABLE_BOTTLE_CROP:
        print("Загрузка BOTTLE_CROP (YOLO11)...", flush=True)
        from preprocessing.bottle_crop import BottleCropProfile
        bottle_crop = BottleCropProfile(output_dir=QUERY_CROP_DIR, weights=str(YOLO_WEIGHTS))
    else:
        print("BOTTLE_CROP выключен (ENABLE_BOTTLE_CROP=False)", flush=True)

    print("Проверка VLM-сервера (vLLM, localhost:8000)...", flush=True)
    try:
        r = requests.get("http://localhost:8000/health", timeout=10)
        r.raise_for_status()
        print("VLM-сервер отвечает", flush=True)
    except Exception as e:
        print(f"ВНИМАНИЕ: VLM-сервер не отвечает ({e}) - запросы будут падать, пока его не поднимут", flush=True)

    print("Загрузка каталожных векторов и нормализатора...", flush=True)
    catalog = np.load(CATALOG_NPZ)
    ref_slugs = list(catalog.keys())
    ref_vecs = np.array([catalog[s] for s in ref_slugs], dtype=np.float64)
    ref_vecs_n = ref_vecs / np.linalg.norm(ref_vecs, axis=1, keepdims=True)

    manifest_rows = list(csv.DictReader(open(CATALOG_MANIFEST, encoding="utf-8")))
    cat_name = {r["slug"]: r["name"] for r in manifest_rows}

    wines = {}
    with open(WINES_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            wines[row["Slug"]] = row

    low_res = {line.split(",")[0] for line in open(LOW_RES_CSV, encoding="utf-8").read().splitlines()[1:]}
    producer_aliases = json.loads(PRODUCER_ALIASES_JSON.read_text(encoding="utf-8"))
    grape_aliases = json.loads(GRAPE_ALIASES_JSON.read_text(encoding="utf-8"))
    with open(TOKEN_DF_PKL, "rb") as f:
        token_df_data = pickle.load(f)

    print("Считаю вкусовые профили (радиальная диаграмма)...", flush=True)
    taste_profiles = {
        slug: _compute_taste_profile(
            w.get("Название вина", ""), slug, w.get("Описание", ""), w.get("Сорт винограда", ""),
        )
        for slug, w in wines.items()
    }

    catalog_list = [
        {
            "slug": slug, "name": w.get("Название вина", slug), "winery": w.get("Винодельня", ""),
            "region": w.get("Регион", ""), "grape": w.get("Сорт винограда", ""),
            "category": w.get("Категория", ""), "color": w.get("Цвет", ""),
            "description": w.get("Описание", ""), "photo": w.get("Фото", ""),
            "low_res": slug in low_res, "taste_profile": taste_profiles[slug],
        }
        for slug, w in wines.items()
    ]

    _state.update({
        "embed_model": embed_model, "bottle_crop": bottle_crop,
        "ref_slugs": ref_slugs, "ref_vecs_n": ref_vecs_n, "cat_name": cat_name, "wines": wines,
        "low_res": low_res, "producer_aliases": producer_aliases, "grape_aliases": grape_aliases,
        "token_df": token_df_data["df"], "token_n_docs": token_df_data["n_docs"],
        "candidate_cache": {}, "catalog_list": catalog_list, "recent_predictions": {},
        "taste_profiles": taste_profiles,
    })
    print(f"Готово за {time.time()-t0:.1f}с — {len(ref_slugs)} референсов, embedding загружен, VLM через vLLM-сервер", flush=True)


def _cached_candidate(slug: str) -> dict:
    cache = _state["candidate_cache"]
    if slug not in cache:
        cache[slug] = load_candidate_fields(slug)
    return cache[slug]


def _card(slug: str) -> dict:
    w = _state["wines"].get(slug, {})
    return {
        "slug": slug,
        "name": w.get("Название вина", _state["cat_name"].get(slug, slug)),
        "winery": w.get("Винодельня", ""),
        "region": w.get("Регион", ""),
        "grape": w.get("Сорт винограда", ""),
        "category": w.get("Категория", ""),
        "color": w.get("Цвет", ""),
        "description": w.get("Описание", ""),
        "photo": w.get("Фото", ""),
        "low_res": slug in _state["low_res"],
        "taste_profile": _state["taste_profiles"].get(slug),
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _find_predict_entry(request_id: str) -> dict | None:
    if request_id in _state["recent_predictions"]:
        return _state["recent_predictions"][request_id]
    for rec in reversed(_read_jsonl(PREDICT_LOG_PATH)):
        if rec["request_id"] == request_id:
            return rec
    return None


def _append_scanner_error_doc(entry: dict, marked_slug: str) -> None:
    top1 = next(c for c in entry["top5"] if c["slug"] == entry["top1_slug"])
    marked = next(c for c in entry["top5"] if c["slug"] == marked_slug)
    marked_rank = entry["top5"].index(marked) + 1
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
    query_photo_rel = f"query_log/{entry['request_id']}.jpg"
    is_new = not SCANNER_ERRORS_MD.exists()
    with open(SCANNER_ERRORS_MD, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "# Ошибки сканера — верный вариант не на 1-м месте\n\n"
                "Автособираемый журнал: человек разметил топ-5 и оказалось, что "
                "верное вино нашлось в топ-5, но не заняло 1-е место (значит "
                "именно 1-е место и уходит наружу как ответ сервиса). Формат и "
                "как этим пользоваться — см. `METRICS.md`.\n\n---\n\n"
            )
        f.write(
            f"## {ts} — request_id `{entry['request_id']}`\n\n"
            f"- Фото запроса: `{query_photo_rel}`\n"
            f"- Предсказано (1-е место, неверно): **{top1['name']}** "
            f"(`{top1['slug']}`) — similarity {top1['similarity']}, "
            f"rerank {top1['rerank_score']}"
            f"{' ⚠ low-res референс' if top1['low_res'] else ''}\n"
            f"- Верно на самом деле (по разметке, {marked_rank}-е место в топ-5): "
            f"**{marked['name']}** (`{marked['slug']}`) — similarity "
            f"{marked['similarity']}, rerank {marked['rerank_score']}"
            f"{' ⚠ low-res референс' if marked['low_res'] else ''}\n\n"
        )


def _append_confident_miss_doc(entry: dict) -> None:
    top1 = next(c for c in entry["top5"] if c["slug"] == entry["top1_slug"])
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
    query_photo_rel = f"query_log/{entry['request_id']}.jpg"
    is_new = not CONFIDENT_MISSES_MD.exists()
    with open(CONFIDENT_MISSES_MD, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "# Уверенные промахи — confidence выше порога, но ни один из топ-5 не подошёл\n\n"
                "Автособираемый журнал: человек разметил топ-5 как «ни один не подошёл», при "
                "этом сервис был уверен (`confidence >= 0.76`, поле `found: true`). Это не "
                "ошибка ранжирования внутри каталога (см. `SCANNER_ERRORS.md` для того случая) "
                "- скорее всего, на фото вино, которого нет в каталоге вообще, но сервис "
                "ошибочно принял его за похожее по фото. Полезно для дальнейшей настройки "
                "порога/реранкинга - см. `METRICS.md`.\n\n---\n\n"
            )
        top5_lines = "\n".join(
            f"  {i+1}. **{c['name']}** (`{c['slug']}`) — similarity {c['similarity']}, "
            f"rerank {c['rerank_score']}{' ⚠ low-res референс' if c['low_res'] else ''}"
            for i, c in enumerate(entry["top5"])
        )
        f.write(
            f"## {ts} — request_id `{entry['request_id']}`\n\n"
            f"- Фото запроса: `{query_photo_rel}`\n"
            f"- Предсказано уверенно (неверно): **{top1['name']}** (`{top1['slug']}`), "
            f"confidence {entry['confidence']}\n"
            f"- Весь топ-5, который был показан и отклонён целиком:\n{top5_lines}\n\n"
        )


@app.get("/health")
def health():
    return {
        "ok": True, "model_loaded": "embed_model" in _state, "catalog_size": len(_state.get("ref_slugs", [])),
        "eval_mode": ENABLE_CONTRACT_ENDPOINT,
    }


@app.get("/catalog")
def catalog():
    return _state["catalog_list"]


@app.get("/v1/similar_by_taste")
def similar_by_taste(slug: str, limit: int = 5, same_winery_quota: int = 2):
    wines = _state["wines"]
    if slug not in wines:
        raise HTTPException(status_code=404, detail="slug not found")
    taste_profiles = _state["taste_profiles"]
    base = wines[slug]
    base_category = base.get("Категория", "")
    base_winery = base.get("Винодельня", "")
    base_region = base.get("Регион", "")
    base_sweetness = (taste_profiles.get(slug) or {}).get("sweetness")

    same_winery_pool, other_pool = [], []
    for s, w in wines.items():
        if s == slug or w.get("Категория", "") != base_category:
            continue
        sw = (taste_profiles.get(s) or {}).get("sweetness")
        if base_sweetness is not None and sw is not None and abs(sw - base_sweetness) > 1:
            continue
        winery = w.get("Винодельня", "")
        if bool(base_winery) and winery == base_winery:
            same_winery_pool.append(s)
        else:
            same_region = bool(base_region) and w.get("Регион", "") == base_region
            other_pool.append((0 if same_region else 1, winery, s))
    other_pool.sort(key=lambda x: x[0])

    picked_same = same_winery_pool[:same_winery_quota]
    remaining_slots = limit - len(picked_same)
    picked_other, seen_wineries = [], set()
    for _, winery, s in other_pool:
        if len(picked_other) >= remaining_slots:
            break
        if winery in seen_wineries:
            continue
        seen_wineries.add(winery)
        picked_other.append(s)

    result_slugs = picked_same + picked_other
    return {"slug": slug, "results": [_card(s) for s in result_slugs]}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _embed_top5(image_path: str) -> tuple[list[dict], float]:
    t0 = time.time()
    vec = _state["embed_model"].encode([image_path])[0]
    vec_n = vec / np.linalg.norm(vec)
    sims = _state["ref_vecs_n"] @ vec_n
    top_idx = np.argsort(-sims)[:TOP_K]
    ref_slugs = _state["ref_slugs"]
    top5 = [{"slug": ref_slugs[i], "similarity": float(sims[i])} for i in top_idx]
    return top5, time.time() - t0


def _process_image_path(src_path: Path) -> dict:
    t0 = time.time()
    prepped_path = str(src_path) + "_prepped.jpg"
    crop_prepped_path = str(src_path) + "_crop_prepped.jpg"
    crop_output_path: str | None = None

    try:
        im, _meta = normalize_for_budget(str(src_path), MIN_PIXELS, MAX_PIXELS, 200, mode="area")
        im.save(prepped_path, "JPEG", quality=95)

        # --- шаг 1: эмбеддинг по raw-фото, TOP-5 ---
        top5_raw, embed_seconds = _embed_top5(prepped_path)

        # --- шаг 1б: BOTTLE_CROP - выключен на GPU-сервере (ENABLE_BOTTLE_CROP=False,
        # см. docstring файла) - код сохранён для будущего включения после
        # отдельной проверки, сейчас всегда используется top5_raw.
        top5_crop = None
        crop_status = "disabled"
        if ENABLE_BOTTLE_CROP and _state.get("bottle_crop") is not None:
            try:
                crop_result = _state["bottle_crop"].process(str(src_path), output_stem=f"query_{uuid.uuid4().hex}")
                crop_status = crop_result.status
                if crop_result.output_path:
                    crop_output_path = crop_result.output_path
                    crop_im, _ = normalize_for_budget(crop_output_path, MIN_PIXELS, MAX_PIXELS, 200, mode="area")
                    crop_im.save(crop_prepped_path, "JPEG", quality=95)
                    top5_crop, embed_seconds_crop = _embed_top5(crop_prepped_path)
                    embed_seconds += embed_seconds_crop
            except Exception as e:
                crop_status = f"error: {e}"

        if top5_crop is not None and top5_crop[0]["similarity"] > top5_raw[0]["similarity"]:
            top5, query_preprocess = top5_crop, "crop"
        else:
            top5, query_preprocess = top5_raw, "raw"

        # --- шаг 2: VLM-JSON на query (через vLLM-сервер, см. _vlm_generate) ---
        t_vlm = time.time()
        vlm_text = _vlm_generate(prepped_path)
        parsed, vlm_json_ok = lenient_parse(vlm_text)
        vlm_seconds = time.time() - t_vlm

        query_fields = {
            "producer": normalize_producer(parsed.get("producer")),
            "grapes": set(normalize_grapes_blend(parsed.get("grape"))),
            "color": extract_color(parsed.get("category")),
            "name_norm": norm_key(_as_text(parsed.get("visible_name"))),
            "raw_tokens": tokens_of(parsed.get("raw_visible_text")),
        }

        # --- шаг 3: реранкинг (IDF + low-res gate) ---
        emb_top1_slug = top5[0]["slug"]
        low_res = _state["low_res"]
        if emb_top1_slug in low_res:
            reranked = [(c["slug"], c["similarity"], c["similarity"], 0.0) for c in top5]
        else:
            reranked = []
            for c in top5:
                cand = _cached_candidate(c["slug"])
                t = text_score(query_fields, cand) / TEXT_MAX
                reranked.append((c["slug"], c["similarity"] + BETA * t, c["similarity"], t))
        reranked.sort(key=lambda x: -x[1])
        top1_slug = reranked[0][0]
        top1_emb_sim = reranked[0][2]

        top1_low_res = top1_slug in low_res

        elapsed = time.time() - t0
        request_id = str(uuid.uuid4())
        top5_cards = [{**_card(s), "similarity": round(sim, 4), "rerank_score": round(score, 4)}
                      for s, score, sim, _t in reranked]
        response = {
            "request_id": request_id,
            "slug": top1_slug,
            "confidence": round(top1_emb_sim, 4),
            "found": top1_emb_sim >= CONFIDENCE_THRESHOLD,
            "card": _card(top1_slug),
            "top5": top5_cards,
            "low_res_warning": top1_low_res,
            "low_res_message": (
                "Исходный референс распознанного вина имеет критически низкое "
                "разрешение. Данные поиска могут быть неточными."
            ) if top1_low_res else None,
            "vlm_extracted": parsed,
            "vlm_json_ok": vlm_json_ok,
            "query_preprocess": query_preprocess,
            "query_preprocess_detail": {
                "raw_top1_similarity": round(top5_raw[0]["similarity"], 4),
                "crop_status": crop_status,
                "crop_top1_similarity": round(top5_crop[0]["similarity"], 4) if top5_crop is not None else None,
            },
            "timing_ms": {"embed": round(embed_seconds * 1000), "vlm": round(vlm_seconds * 1000),
                          "total": round(elapsed * 1000)},
        }

        try:
            shutil.copyfile(prepped_path, QUERY_LOG_DIR / f"{request_id}.jpg")
        except OSError:
            pass
        log_entry = {
            "request_id": request_id, "timestamp": time.time(),
            "top1_slug": top1_slug, "confidence": response["confidence"], "found": response["found"],
            "top5": top5_cards, "query_preprocess": query_preprocess,
        }
        _append_jsonl(PREDICT_LOG_PATH, log_entry)
        _state["recent_predictions"][request_id] = log_entry

        return response
    finally:
        Path(prepped_path).unlink(missing_ok=True)
        Path(crop_prepped_path).unlink(missing_ok=True)
        if crop_output_path:
            Path(crop_output_path).unlink(missing_ok=True)


_inference_lock = threading.Lock()


async def _run_pipeline(image: UploadFile) -> dict:
    with tempfile.NamedTemporaryFile(suffix=Path(image.filename or "query.jpg").suffix or ".jpg", delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)
    try:
        with _inference_lock:
            return _process_image_path(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


if ENABLE_CONTRACT_ENDPOINT:
    @app.post("/v1/eval/predict")
    async def predict_eval(image: UploadFile = File(...)):
        result = await _run_pipeline(image)
        return JSONResponse({"slug": result["slug"]})

    @app.post("/v1/topk")
    async def predict_topk(image: UploadFile = File(...)):
        response = await _run_pipeline(image)
        return JSONResponse([
            {"slug": c["slug"], "confidence": c["similarity"]}
            for c in response["top5"]
        ])


EVAL_MODE_BUSY_MESSAGE = (
    "Пакетная обработка временно недоступна — включён контрактный режим "
    "внешней проверки, а пачка фото надолго занимает единственную модель "
    "и может сорвать таймаут проверки. Одиночное распознавание работает как обычно."
)


@app.post("/v1/predict")
async def predict_personal(image: UploadFile = File(...), x_api_key: str = Header(default="")):
    # Одиночное распознавание работает всегда, в том числе при включённом
    # контрактном эндпоинте: доступ к GPU сериализуется _inference_lock, один
    # запрос сайта (~3с) + контрактный запрос укладываются в его лимит 10с.
    # Блокируется только пакетная обработка (см. /v1/batch/run) — вот она
    # держит модель до минуты и SLA проверки сорвать может.
    if x_api_key != PERSONAL_API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")
    return JSONResponse(await _run_pipeline(image))


class BatchRunRequest(BaseModel):
    batch_id: str


@app.post("/v1/batch/upload")
async def batch_upload(batch_id: str = Form(...), image: UploadFile = File(...)):
    if not re.fullmatch(r"[a-zA-Z0-9-]{8,64}", batch_id):
        raise HTTPException(status_code=400, detail="некорректный batch_id")
    batch_dir = BATCH_STAGING_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(image.filename or f"{uuid.uuid4()}.jpg").name
    dest = batch_dir / safe_name
    try:
        content = await image.read()
        dest.write_bytes(content)
    except OSError as e:
        return {"ok": False, "filename": safe_name, "error": str(e)}
    return {"ok": True, "filename": safe_name}


def _run_batch_worker(batch_id: str, files: list[Path]) -> None:
    state = _batches[batch_id]
    for i, path in enumerate(files):
        try:
            with _inference_lock:
                result = _process_image_path(path)
            state["items"][i] = {"filename": path.name, "status": "done", "result": result}
        except Exception as e:  # noqa: BLE001
            state["items"][i] = {"filename": path.name, "status": "error", "result": None, "error": str(e)}
        state["done"] = i + 1
    state["status"] = "finished"


@app.post("/v1/batch/run")
def batch_run(body: BatchRunRequest):
    if ENABLE_CONTRACT_ENDPOINT:
        raise HTTPException(status_code=503, detail=EVAL_MODE_BUSY_MESSAGE)
    batch_dir = BATCH_STAGING_DIR / body.batch_id
    if not batch_dir.is_dir():
        raise HTTPException(status_code=404, detail="batch_id не найден - сначала залейте файлы через /v1/batch/upload")
    existing = _batches.get(body.batch_id)
    if existing and existing["status"] == "running":
        raise HTTPException(status_code=409, detail="эта пачка уже обрабатывается")
    files = sorted(p for p in batch_dir.iterdir() if p.is_file())
    if not files:
        raise HTTPException(status_code=400, detail="в пачке нет файлов")
    _batches[body.batch_id] = {
        "status": "running", "total": len(files), "done": 0,
        "items": [{"filename": p.name, "status": "pending", "result": None} for p in files],
    }
    threading.Thread(target=_run_batch_worker, args=(body.batch_id, files), daemon=True).start()
    return {"ok": True, "batch_id": body.batch_id, "total": len(files)}


@app.get("/v1/batch/status/{batch_id}")
def batch_status(batch_id: str):
    state = _batches.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="batch_id не найден")
    return state


class MarkRequest(BaseModel):
    request_id: str
    marked_slug: str | None = None
    # True — человек подтвердил, что этого вина нет в каталоге вообще (ground
    # truth), не просто "не оказалось в топ-5". Разные вещи: первое — не
    # ошибка распознавания, второе — она. Смешивать их в одном знаменателе
    # точности некорректно (см. SCALABILITY.md/README про battle_photos —
    # обычно найдено экспериментально 28.09.2026: без этого флага панель
    # статистики топила настоящую точность распознавания среди тестов на
    # заведомо отсутствующих в каталоге винах).
    not_in_catalog: bool = False


@app.post("/v1/mark")
def mark(body: MarkRequest):
    entry = _find_predict_entry(body.request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="request_id не найден в журнале предсказаний")

    top1_slug = entry["top1_slug"]
    top5_slugs = [c["slug"] for c in entry["top5"]]
    not_in_catalog = body.not_in_catalog
    is_top1_correct = (not not_in_catalog) and body.marked_slug == top1_slug
    is_in_top5 = (not not_in_catalog) and bool(body.marked_slug) and body.marked_slug in top5_slugs
    is_scanner_error = is_in_top5 and not is_top1_correct
    is_confident_miss = body.marked_slug is None and bool(entry.get("found"))

    mark_record = {
        "request_id": body.request_id, "timestamp": time.time(),
        "top1_slug": top1_slug, "marked_slug": body.marked_slug,
        "not_in_catalog": not_in_catalog,
        "is_top1_correct": is_top1_correct, "is_in_top5": is_in_top5,
        "is_scanner_error": is_scanner_error, "is_confident_miss": is_confident_miss,
    }
    _append_jsonl(MARKS_LOG_PATH, mark_record)
    if is_scanner_error:
        _append_scanner_error_doc(entry, body.marked_slug)
    if is_confident_miss:
        _append_confident_miss_doc(entry)

    return {
        "ok": True, "is_top1_correct": is_top1_correct,
        "is_in_top5": is_in_top5, "is_scanner_error": is_scanner_error,
        "is_confident_miss": is_confident_miss,
    }


@app.get("/v1/metrics_doc")
def metrics_doc():
    return {"content": METRICS_MD_PATH.read_text(encoding="utf-8")}


@app.get("/v1/stats")
def stats():
    """Внутренняя статистика по ручной разметке (для dev-режима UI).

    Точность считается только по разметкам, где вино есть в каталоге
    (not_in_catalog=False) — разметки заведомо отсутствующих в каталоге вин
    не являются ошибкой распознавания и не занижают точность в том же
    знаменателе."""
    total_comparisons = _count_lines(PREDICT_LOG_PATH)
    marks = _read_jsonl(MARKS_LOG_PATH)
    total_marked = len(marks)
    catalog_marks = [m for m in marks if not m.get("not_in_catalog")]
    not_in_catalog_marked = total_marked - len(catalog_marks)
    catalog_marked = len(catalog_marks)
    tp = sum(1 for m in catalog_marks if m["is_top1_correct"])
    top5_hits = sum(1 for m in catalog_marks if m["is_in_top5"])
    scanner_errors = sum(1 for m in catalog_marks if m["is_scanner_error"])
    confident_misses = sum(1 for m in catalog_marks if m.get("is_confident_miss"))
    confident_misses_not_in_catalog = sum(
        1 for m in marks if m.get("not_in_catalog") and m.get("is_confident_miss")
    )
    top1_accuracy = round(tp / catalog_marked, 4) if catalog_marked else None
    top5_recall = round(top5_hits / catalog_marked, 4) if catalog_marked else None
    return {
        "total_comparisons": total_comparisons,
        "total_marked": total_marked,
        "catalog_marked": catalog_marked,
        "not_in_catalog_marked": not_in_catalog_marked,
        "top1_correct": tp,
        "top1_accuracy": top1_accuracy,
        "top5_recall": top5_recall,
        "scanner_errors": scanner_errors,
        "confident_misses": confident_misses,
        "confident_misses_not_in_catalog": confident_misses_not_in_catalog,
    }
