# Запуск и эксплуатация

## Требования

- Linux с NVIDIA GPU (CUDA). Проверено на RTX 4090; достаточно ~16–24 ГБ
  VRAM, на меньших картах снизить `--gpu-memory-utilization`.
- Docker + `nvidia-container-toolkit` (Docker-путь) либо Python 3.12
  (ручной путь).
- Несколько гигабайт под веса моделей + каталог (`data/catalog/`,
  ~259 МБ, уже в репозитории).
- Интернет для загрузки моделей с Hugging Face при первом старте.

## Docker

```bash
cp .env.example .env
docker compose up
```

Поднимает два контейнера:
- `vllm` — официальный образ `vllm/vllm-openai:v0.30.0`, читающая модель;
- `winescanner` — собирается из `Dockerfile`, визуальный поиск + API.

Разделение на два образа обязательно: vLLM и `sentence-transformers`
требуют несовместимых версий torch, в одно окружение не ставятся.
`docker-compose.yml` связывает их через `depends_on` + `healthcheck`.

Проверка доступа GPU из контейнера:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Флаг `--enforce-eager` у vLLM

В `docker-compose.yml` у vLLM выставлен `--enforce-eager` — отключает
JIT-компиляцию движка (`torch.compile`). Контейнер поднимается за ~100 с
предсказуемо, без прогрева кэша компиляции. Цена — инференс на несколько
процентов медленнее на запрос; в лимит 10 с укладывается с запасом.

Для максимальной скорости инференса убрать `--enforce-eager` и
примонтировать постоянный volume под `/root/.cache/vllm`: первый старт
будет долгим (полная компиляция), последующие — быстрыми за счёт кэша.

### DNS в сборке

Если хост резолвит DNS через `127.0.0.53` (Ubuntu с systemd-resolved),
buildkit-сборка (`docker compose build`) не получает рабочий DNS и
`apt-get` внутри `RUN` зависает. Задать явные DNS-серверы на хосте:

```json
// /etc/docker/daemon.json
{ "dns": ["8.8.8.8", "1.1.1.1"] }
```
```bash
systemctl restart docker
```

## Ручная установка (без Docker)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# torch/torchvision под CUDA 12.1 — отдельным индексом пакетов
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
cp .env.example .env
```

Системный CUDA-тулкит не требуется: torch-колёса с индекса cu121 несут
собственные `nvidia-*-cu12` библиотеки. Нужен только видеодрайвер хоста.

### Запуск

Два процесса, порядок важен — сначала vLLM, затем сервис:

```bash
# 1) vLLM-сервер (адрес — WINESCANNER_VLLM_URL, по умолчанию localhost:8000)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.65 --port 8000 --trust-remote-code --enforce-eager

# 2) сам сервис
uvicorn winescanner.app:app --host 0.0.0.0 --port 8080
```

Готовность:

```bash
curl -s http://127.0.0.1:8080/health
# {"ok": true, "model_loaded": true, "catalog_size": 2103, "eval_mode": false}
```

## Продовый запуск через systemd

Два юнита; `winescanner.service` зависит от `vllm.service`.
`KillMode=control-group` убивает весь cgroup процесса при stop/restart.

```ini
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM OpenAI-compatible server (Qwen3-VL-4B-Instruct)
After=network.target

[Service]
Type=simple
ExecStart=/path/to/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.65 --port 8000 --trust-remote-code --enforce-eager
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/winescanner.service
[Unit]
Description=Wine Scanner service (FastAPI + uvicorn)
After=network.target vllm.service
Requires=vllm.service

[Service]
Type=simple
WorkingDirectory=/path/to/repo
ExecStart=/path/to/venv/bin/uvicorn winescanner.app:app --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=5
KillMode=control-group
TimeoutStopSec=15

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now vllm.service winescanner.service
journalctl -u winescanner -f
```

`systemctl restart winescanner` перечитывает `.env` при старте процесса.

## Переменные окружения

Читаются из `.env` (шаблон — `.env.example`). Без `.env` — значения по
умолчанию.

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `WINESCANNER_API_KEY` | Ключ (`X-API-Key`) для `/v1/predict` | заглушка, **задать своё** |
| `WINESCANNER_ENABLE_CONTRACT_ENDPOINT` | Регистрирует `/v1/eval/predict`, `/v1/topk` | `true` (в `.env.example`) |
| `WINESCANNER_BASE_DIR` | Корень с данными `data/catalog/` | папка репозитория |
| `WINESCANNER_VLLM_URL` | Адрес vLLM-сервера | `http://localhost:8000/v1/chat/completions` |
| `WINESCANNER_CONFIDENCE_THRESHOLD` | Порог поля `found` | `0.76` |
| `WINESCANNER_TOP_K` | Число кандидатов | `5` |
| `WINESCANNER_ENABLE_BOTTLE_CROP` | Экспериментальный кроп бутылки (YOLO) | `false` |

## Контрактный эндпоинт

| `WINESCANNER_ENABLE_CONTRACT_ENDPOINT` | `/v1/eval/predict`, `/v1/topk` | `/v1/predict` | `/v1/batch/run` |
|---|---|---|---|
| `false` | `404` | работает | работает |
| `true` | работает | работает | `503` |

Одиночное распознавание и контрактный эндпоинт работают одновременно:
доступ к модели сериализуется, оба укладываются в лимит 10 с.
Блокируется только пакетная обработка — она держит модель до минуты.
Ответ `/v1/eval/predict` урезан до `{"slug": "..."}`. Текущее значение
флага видно в `/health` (поле `eval_mode`).

## Логи и журналы

Внутри `winescanner/`, относительно `WINESCANNER_BASE_DIR`; в git не
попадают (`.gitignore`), создаются автоматически.

| Файл/папка | Содержимое |
|---|---|
| `predict_log.jsonl` | все вызовы `/v1/predict` |
| `marks_log.jsonl` | разметка через `/v1/mark` |
| `SCANNER_ERRORS.md` | верный вариант был в топ-5, но не на 1-м месте |
| `CONFIDENT_MISSES.md` | уверенный и полностью неверный ответ |
| `query_log/` | сохранённые фото запросов, по файлу на `request_id` |
| `batch_staging/<batch_id>/` | оригиналы файлов пачки (состояние пачки — в памяти процесса) |

## Эксплуатационные особенности

- **Один запрос за раз.** Одна копия каждой модели на GPU; конкурентные
  вызовы сериализуются, а не выполняются параллельно.
- **Фронтенд без сборки.** Веб-страница — один файл
  `winescanner/static/index.html`, читается с диска на каждый запрос `/`;
  правки видны без перезапуска сервиса.
