# Образ сервиса winescanner (FastAPI + эмбеддинг-модель на torch).
# Читающая модель (VLM) — отдельный контейнер на официальном образе vLLM,
# см. docker-compose.yml (причина разделения — несовместимые версии torch).
#
# База — ubuntu:24.04 (Python 3.12 из коробки), без системного
# CUDA-тулкита: torch-колёса с индекса cu121 несут собственные CUDA-
# библиотеки как pip-зависимости; нужен только видеодрайвер хоста через
# nvidia-container-toolkit.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision под CUDA 12.1 — отдельный индекс пакетов (не PyPI).
RUN pip install --no-cache-dir --break-system-packages torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Модули, которые импортирует winescanner/app.py
COPY prompt_v4.py .
COPY retrievers/ retrievers/
COPY preprocessing/ preprocessing/
COPY winescanner/ winescanner/
# docs/METRICS.md отдаётся эндпоинтом /v1/metrics_doc
COPY docs/ docs/

# Данные каталога (~259 МБ) запекаются в образ — см. data/README.md.
COPY data/ data/

ENV WINESCANNER_BASE_DIR=/app

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "winescanner.app:app", "--host", "0.0.0.0", "--port", "8080"]
