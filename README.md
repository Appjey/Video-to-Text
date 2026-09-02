# Vide-to-Text

Транскрибация видео и аудио в текст с помощью Whisper. Результат сохраняется рядом с исходным файлом в виде `.txt`.

## Структура проекта

| Файл | Назначение |
|------|------------|
| `main.py` | Основной скрипт: Whisper large-v3 через `transformers`. Запускается из venv. |
| `stt_cli.py` | Переносимая CLI-утилита на `faster-whisper` (VAD, диаризация, откат на CPU). Из неё собирается `stt_cli.exe`. |
| `stt_cli.spec` | Конфигурация PyInstaller для сборки `stt_cli.exe`. |
| `.github/workflows/release.yml` | CI: сборка `stt_cli.exe` под Windows и публикация релиза. |
| `extras/audio_stream.py` | Эксперимент: транскрибация с микрофона в реальном времени. |
| `extras/answering.py` | Эксперимент: выжимка договорённостей из готовой транскрипции через LLM. |
| `mnt/data/` | Локальные медиафайлы и результаты (в git не попадают). |

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Оба скрипта используют `ffmpeg`: `main.py` ищет его в PATH, иначе берёт бинарник из пакета `imageio-ffmpeg`;
`stt_cli.py` ищет его в PATH либо рядом со скриптом (`./ffmpeg/ffmpeg.exe`).

## main.py

```bash
python main.py "mnt/data/запись.mp4"
python main.py "mnt/data/звонок.amr" --language ru --output результат.txt
```

Параметры:

- `input` — путь к видео/аудио (mp4, mkv, mp3, wav, amr, m4a ...);
- `-o, --output` — куда сохранить текст (по умолчанию рядом: `<имя>_transcription.txt`);
- `--language` — код языка, по умолчанию `ru`;
- `--model` — модель Whisper с Hugging Face, по умолчанию `openai/whisper-large-v3`.

## stt_cli.py / stt_cli.exe

Готовый `stt_cli.exe` можно взять из раздела Releases: в архив уже вложены `ffmpeg` и CUDA-библиотеки.

```bash
./stt_cli.exe "./2025-08-19 15-53-16.mkv" --device cpu
python stt_cli.py "запись.mkv" --language ru --vad
```

Ключевые параметры: `--model`, `--device auto|cuda|cpu`, `--language`, `--vad`, `--diarize --hf-token ...`, `--gpu-required`.
Полный список: `python stt_cli.py --help`.

Сборка exe локально:

```bash
pip install pyinstaller==6.10.0
pyinstaller stt_cli.spec
```
