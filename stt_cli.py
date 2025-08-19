#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import shutil
import subprocess
import tempfile
import time
import platform
import math
from typing import List, Tuple, Optional

try:
    from faster_whisper import WhisperModel
except Exception as e:
    print("Не удалось импортировать faster_whisper. Установите зависимости из requirements.txt.")
    raise

# --------------------------
# УТИЛИТЫ
# --------------------------

def find_ffmpeg() -> str:
    """
    Ищем ffmpeg в PATH или рядом с исполняемым файлом (упаковано в bundle).
    Поддерживаем PyInstaller (sys._MEIPASS) и относительный каталог ./third_party/ffmpeg/bin.
    """
    ffmpeg_name = "ffmpeg.exe" if platform.system().lower().startswith("win") else "ffmpeg"

    # 1) PATH
    path_ffmpeg = shutil.which(ffmpeg_name)
    if path_ffmpeg:
        return path_ffmpeg

    # 2) Рядом с exe (PyInstaller)
    base_dirs = []
    if getattr(sys, "_MEIPASS", None):
        base_dirs.append(os.path.join(sys._MEIPASS))  # PyInstaller temp dir

    # 3) Локальные каталоги проекта
    exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))
    base_dirs += [
        exe_dir,
        os.path.join(exe_dir, "ffmpeg"),
        os.path.join(exe_dir, "third_party", "ffmpeg", "bin"),
        os.path.join(exe_dir, "bin"),
    ]

    for base in base_dirs:
        candidate = os.path.join(base, ffmpeg_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "ffmpeg не найден. Добавьте ffmpeg в PATH или положите бинарник рядом с программой в ./third_party/ffmpeg/bin/"
    )

def extract_audio_to_wav16k_mono(input_media_path: str, ffmpeg_path: str) -> str:
    """
    Извлекаем/конвертируем аудио в WAV PCM16 16kHz mono (максимально совместимо).
    Возвращаем путь к временному WAV.
    """
    tmp_dir = tempfile.mkdtemp(prefix="stt_cli_")
    wav_path = os.path.join(tmp_dir, "audio_16k_mono.wav")

    cmd = [
        ffmpeg_path, "-y",
        "-i", input_media_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not os.path.exists(wav_path):
        raise RuntimeError(f"ffmpeg не смог извлечь аудио:\n{proc.stderr}")
    return wav_path

def normalize_space(text: str) -> str:
    # Убираем двойные пробелы/переносы, приводим к аккуратному виду
    t = " ".join(text.replace("\r", "\n").split())
    return t.strip()

def wrap_by_sentences(text: str) -> List[str]:
    """
    Очень простая разбивка по предложениям — точка/воскл/вопрос.
    Для «умной» сегментации можно позже добавить razdel/spacy, но они утяжеляют сборку.
    """
    if not text:
        return []
    out = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in ".?!…":
            seg = "".join(buf).strip()
            if seg:
                out.append(seg)
            buf = []
    rest = "".join(buf).strip()
    if rest:
        out.append(rest)
    return out

def format_timestamp(seconds: float) -> str:
    # Используется только для логов прогресса (если надо)
    if seconds < 0:
        seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# --------------------------
# ДИАРИЗАЦИЯ (ОПЦИОНАЛЬНО)
# --------------------------

def run_diarization(
    wav_path: str,
    hf_token: Optional[str],
    num_speakers: Optional[int]
):
    """
    Возвращает список спикер-сегментов [(start, end, label), ...].
    Требует pyannote.audio и модели pyannote/speaker-diarization-3.1 (нужен HF токен).
    Если что-то идёт не так — бросаем исключение (выше обработаем и продолжим без диаризации).
    """
    from pyannote.audio import Pipeline

    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    if not hf_token:
        raise RuntimeError("Для диаризации нужен --hf-token (Hugging Face) для доступа к моделям pyannote.")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline({"audio": wav_path}, **kwargs)

    # Преобразуем в плоский список
    segments = []
    # diarization.itertracks(yield_label=True) -> (segment, track, label)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        label = str(speaker)
        segments.append((start, end, label))

    # Сортируем по времени
    segments.sort(key=lambda x: (x[0], x[1]))
    return segments

def assign_speakers_to_whisper_segments(
    whisper_segments: List[Tuple[float, float, str]],
    speaker_segments: List[Tuple[float, float, str]],
    default_label: str = "SPK00"
) -> List[Tuple[float, float, str, str]]:
    """
    Для каждого сегмента Whisper находим спикера с максимальным пересечением по времени.
    Возвращает [(start, end, text, speaker_label), ...]
    """
    if not speaker_segments:
        return [(s, e, t, default_label) for (s, e, t) in whisper_segments]

    out = []
    i = 0
    n = len(speaker_segments)
    for (ws, we, wt) in whisper_segments:
        best_label = default_label
        best_overlap = 0.0

        # продвигаем указатель
        while i + 1 < n and speaker_segments[i+1][1] <= ws:
            i += 1

        # проверяем несколько ближайших окон
        for j in range(max(0, i-3), min(n, i+4)):
            ss, se, lab = speaker_segments[j]
            # пересечение отрезков [ws,we] и [ss,se]
            left = max(ws, ss)
            right = min(we, se)
            overlap = max(0.0, right - left)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = lab

        out.append((ws, we, wt, best_label))
    return out

# --------------------------
# СКЛЕЙКА В АБЗАЦЫ (TXT)
# --------------------------

def merge_segments_to_paragraphs(
    segments: List[Tuple[float, float, str]],
    join_threshold: float = 0.8,
    max_paragraph_chars: int = 1200
) -> List[str]:
    """
    Склеиваем соседние сегменты, если пауза между ними <= join_threshold.
    Ограничиваем размер абзаца max_paragraph_chars, стараясь резать по границам предложений.
    """
    if not segments:
        return []

    merged: List[str] = []
    cur_text = []
    cur_end = segments[0][1]

    def flush():
        nonlocal cur_text
        block = normalize_space(" ".join(cur_text))
        if not block:
            cur_text = []
            return
        # Если слишком длинно — разрежем по предложениям
        if len(block) > max_paragraph_chars:
            sents = wrap_by_sentences(block)
            buf = []
            for s in sents:
                if len(" ".join(buf + [s])) <= max_paragraph_chars:
                    buf.append(s)
                else:
                    merged.append(normalize_space(" ".join(buf)))
                    buf = [s]
            if buf:
                merged.append(normalize_space(" ".join(buf)))
        else:
            merged.append(block)
        cur_text = []

    prev_end = segments[0][1]
    cur_text.append(segments[0][2])

    for (s, e, t) in segments[1:]:
        gap = s - prev_end
        if gap <= join_threshold and len(" ".join(cur_text + [t])) <= max_paragraph_chars:
            cur_text.append(t)
        else:
            flush()
            cur_text = [t]
        prev_end = e
        cur_end = e

    flush()
    return merged

def merge_segments_by_speaker(
    labeled_segments: List[Tuple[float, float, str, str]],
    speaker_prefix: str = "SPK",
    join_threshold: float = 0.8,
    max_paragraph_chars: int = 1200
) -> List[str]:
    """
    Склеиваем по спикеру: пока спикер не меняется и пауза <= join_threshold — дописываем в тот же абзац.
    Возвращаем список строк: "SPK01: текст..."
    """
    if not labeled_segments:
        return []

    out: List[str] = []
    cur_label = labeled_segments[0][3]
    cur_text = [labeled_segments[0][2]]
    prev_end = labeled_segments[0][1]

    def flush(label: str, buf: List[str]):
        if not buf:
            return
        text = normalize_space(" ".join(buf))
        if not text:
            return
        # Ограничим по длине, разрезая по предложениям
        if len(text) <= max_paragraph_chars:
            out.append(f"{format_spk(label, speaker_prefix)}: {text}")
        else:
            sents = wrap_by_sentences(text)
            para = []
            for s in sents:
                if len(" ".join(para + [s])) <= max_paragraph_chars:
                    para.append(s)
                else:
                    out.append(f"{format_spk(label, speaker_prefix)}: {normalize_space(' '.join(para))}")
                    para = [s]
            if para:
                out.append(f"{format_spk(label, speaker_prefix)}: {normalize_space(' '.join(para))}")

    for (s, e, t, lab) in labeled_segments[1:]:
        gap = s - prev_end
        if lab == cur_label and gap <= join_threshold:
            if len(" ".join(cur_text + [t])) <= max_paragraph_chars:
                cur_text.append(t)
            else:
                flush(cur_label, cur_text)
                cur_text = [t]
        else:
            flush(cur_label, cur_text)
            cur_label = lab
            cur_text = [t]
        prev_end = e

    flush(cur_label, cur_text)
    return out

def format_spk(label: str, prefix: str) -> str:
    """
    Приводим произвольные метки к виду PREFIXNN (например SPK01).
    В pyannote метки бывают 'SPEAKER_00' и т.п.
    """
    # Выделим последние цифры, иначе просто хэш по числу спикеров не нужен — упростим:
    digits = "".join(ch for ch in label if ch.isdigit())
    if digits == "":
        return f"{prefix}00"
    try:
        num = int(digits)
    except ValueError:
        return f"{prefix}00"
    return f"{prefix}{num:02d}"

# --------------------------
# ОСНОВНОЙ PIPELINE
# --------------------------

def detect_default_compute(device: str) -> str:
    """
    Автовыбор compute_type. Для CUDA — float16, для CPU — int8.
    """
    if device == "cuda":
        return "float16"
    return "int8"

def main():
    parser = argparse.ArgumentParser(
        description="Переносимая утилита транскрибации (faster-whisper + CUDA). Вывод — чистый TXT."
    )
    parser.add_argument("input", help="Путь к видео/аудио файлу (mp4, mkv, mp3, wav, ...)")
    parser.add_argument(
        "-o", "--output",
        help="Путь к результату (по умолчанию: рядом с входом, .txt)",
        default=None
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Модель faster-whisper (tiny, base, small, medium, large-v3)"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Локальная папка с моделью (для офлайна)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Устройство"
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Тип вычислений: float16, float32, int8, int8_float16 и т.д."
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Код языка (например: ru, en). Если не задан — автоопределение."
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="transcribe — оставить исходный язык, translate — перевод на английский"
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Включить VAD-фильтр faster-whisper (лучшее разбиение)."
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=5,
        help="best_of (для sampling)."
    )
    parser.add_argument(
        "--join-threshold",
        type=float,
        default=0.8,
        help="Макс. пауза (сек) между сегментами для склейки в один абзац."
    )
    parser.add_argument(
        "--max-paragraph-chars",
        type=int,
        default=1200,
        help="Ограничение длины абзаца (символов) с мягким разбиением по предложениям."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не печатать прогресс по сегментам."
    )

    # Диаризация (опционально)
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Включить разметку по спикерам (pyannote, нужен --hf-token)."
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face токен для загрузки моделей pyannote."
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="Подсказать числом спикеров (если известно)."
    )
    parser.add_argument(
        "--speaker-prefix",
        default="SPK",
        help="Префикс меток спикеров (по умолчанию SPK)."
    )

    args = parser.parse_args()

    ffmpeg_path = find_ffmpeg()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    # Куда сохраняем результат
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if args.output is None:
        output_path = os.path.join(base_dir, f"{base_name}_transcript.txt")
    else:
        output_path = os.path.abspath(args.output)

    # Подготовим WAV 16k mono (надёжно для любых контейнеров)
    wav_path = extract_audio_to_wav16k_mono(input_path, ffmpeg_path)

    # Настройка устройства/типа вычислений
    device = args.device
    if device == "auto":
        device = "cuda" if shutil.which("nvidia-smi") else "cpu"

    compute_type = args.compute_type or detect_default_compute(device)

    # Где хранить/брать модели
    model_source = args.model
    if args.model_dir:
        model_source = os.path.abspath(args.model_dir)

    print(f"⚙️  Параметры:\n"
          f"    input         : {input_path}\n"
          f"    tmp wav       : {wav_path}\n"
          f"    output        : {output_path}\n"
          f"    model         : {args.model}\n"
          f"    model_source  : {model_source}\n"
          f"    device        : {device}\n"
          f"    compute_type  : {compute_type}\n"
          f"    language      : {args.language or 'auto'}\n"
          f"    task          : {args.task}\n"
          f"    vad           : {args.vad}\n"
          f"    beam_size     : {args.beam_size}\n"
          f"    best_of       : {args.best_of}\n"
          f"    join_threshold: {args.join_threshold}s\n"
          f"    max_par_chars : {args.max_paragraph_chars}\n"
          f"    diarize       : {args.diarize}\n"
          f"    speakers_hint : {args.speakers or 'auto'}\n"
          f"    spk_prefix    : {args.speaker_prefix}\n")

    t0 = time.time()

    # 1) Транскрипция
    model = WhisperModel(
        model_source,
        device=device,
        compute_type=compute_type
    )

    segments_iter, info = model.transcribe(
        wav_path,
        task=args.task,
        language=args.language,
        vad_filter=args.vad,
        beam_size=args.beam_size,
        best_of=args.best_of,
        word_timestamps=False
    )

    whisper_segments: List[Tuple[float, float, str]] = []
    idx = 0
    for seg in segments_iter:
        idx += 1
        text = seg.text.strip()
        start = float(seg.start)
        end = float(seg.end)
        whisper_segments.append((start, end, text))
        if not args.quiet:
            print(f"[{idx:04d}] {format_timestamp(start)} → {format_timestamp(end)} | {text}")

    # 2) Опциональная диаризация
    diar_ok = False
    diar_segments: List[Tuple[float, float, str]] = []
    if args.diarize:
        try:
            diar_segments = run_diarization(
                wav_path=wav_path,
                hf_token=args.hf_token,
                num_speakers=args.speakers
            )
            diar_ok = len(diar_segments) > 0
        except Exception as e:
            print(f"Диаризация недоступна: {e}")
            diar_ok = False

    # 3) Формирование TXT вывода
    if diar_ok:
        # Присваиваем спикеров whisper-сегментам, затем склеиваем по спикерам
        labeled = assign_speakers_to_whisper_segments(whisper_segments, diar_segments, default_label=f"{args.speaker_prefix}00")
        paragraphs = merge_segments_by_speaker(
            labeled_segments=labeled,
            speaker_prefix=args.speaker_prefix,
            join_threshold=args.join_threshold,
            max_paragraph_chars=args.max_paragraph_chars
        )
    else:
        # Просто склеиваем в читабельные абзацы без таймкодов/спикеров
        paragraphs = merge_segments_to_paragraphs(
            segments=whisper_segments,
            join_threshold=args.join_threshold,
            max_paragraph_chars=args.max_paragraph_chars
        )

    # 4) Сохранение TXT
    final_text = "\n\n".join(paragraphs).strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"✅ TXT сохранён: {output_path}")

    print(f"⏱️  Готово за {time.time() - t0:.2f} c")

    # 5) Чистим временный WAV
    try:
        os.remove(wav_path)
        os.rmdir(os.path.dirname(wav_path))
    except Exception:
        pass

if __name__ == "__main__":
    main()
