#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Транскрибация видео/аудио в текст: Whisper (transformers) + разметка по спикерам (pyannote.audio).

Примеры:
    python main.py "mnt/data/запись.mp4"
    python main.py "mnt/data/звонок.amr" --speakers 2
    python main.py "лекция.mkv" --no-diarize --no-timestamps --output лекция.txt

Результат сохраняется рядом с исходным файлом как <имя файла>_transcription.txt:

    [00:00:03 - 00:00:09] SPEAKER_00: Добрый день, ...
    [00:00:09 - 00:00:15] SPEAKER_01: Здравствуйте, ...

Как это работает:
  * ffmpeg декодирует звуковую дорожку прямо в память (16 кГц, моно), промежуточных файлов нет;
  * Whisper обрабатывает запись штатным long-form алгоритмом: скользящее 30-секундное окно,
    границы сегментов и таймкоды предсказывает сама модель, при сомнительном результате
    включается temperature fallback (рекомендации из карточки openai/whisper-large-v3);
  * pyannote.audio размечает, кто и когда говорит; сегментам Whisper присваивается спикер
    по максимальному перекрытию, соседние реплики одного спикера склеиваются.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as hf_logging

DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_LANGUAGE = "ru"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
SAMPLE_RATE = 16000          # Whisper и pyannote ждут 16 кГц, моно
WHISPER_WINDOW_SEC = 30      # размер окна Whisper

# Рекомендованные параметры long-form генерации из карточки модели
LONG_FORM_GENERATE_KWARGS = dict(
    condition_on_prev_tokens=False,
    compression_ratio_threshold=1.35,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


# -----------------------------------------------------------------------------
# АУДИО
# -----------------------------------------------------------------------------

def find_ffmpeg() -> str:
    """Ищет ffmpeg в PATH, иначе берёт бинарник из пакета imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise FileNotFoundError("ffmpeg не найден. Добавьте его в PATH или установите пакет imageio-ffmpeg.")


def decode_audio(input_path: str, ffmpeg_path: str) -> np.ndarray:
    """
    Декодирует звуковую дорожку любого видео/аудио (mp4, mkv, m4a, amr, mp3, wav ...)
    сразу в память: float32, 16 кГц, моно. Файлы на диск не пишутся.
    """
    cmd = [
        ffmpeg_path, "-nostdin", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg не смог прочитать файл:\n" + proc.stderr.decode("utf-8", errors="replace"))
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("В файле не найдена звуковая дорожка.")
    return audio


# -----------------------------------------------------------------------------
# WHISPER
# -----------------------------------------------------------------------------

def load_whisper(model_name: str, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, use_safetensors=True)
    model.to(device).eval()
    return processor, model


def transcribe(audio: np.ndarray, processor, model, language: str, device: str) -> List[Segment]:
    """
    Транскрибирует всю запись целиком. Для записей длиннее 30 с transformers использует
    последовательный long-form алгоритм Whisper: модель сама предсказывает таймкоды,
    следующее окно начинается с последнего уверенного таймкода.
    """
    duration = len(audio) / SAMPLE_RATE
    if duration <= WHISPER_WINDOW_SEC:
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        attention_mask = None
    else:
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           truncation=False, padding="longest", return_attention_mask=True)
        attention_mask = inputs.attention_mask.to(device)

    features = inputs.input_features.to(device, dtype=model.dtype)

    with torch.inference_mode():
        out = model.generate(
            features,
            attention_mask=attention_mask,
            language=language,
            task="transcribe",
            return_timestamps=True,
            return_segments=True,
            **LONG_FORM_GENERATE_KWARGS,
        )

    decode = lambda tokens: processor.tokenizer.decode(tokens, skip_special_tokens=True)
    return build_segments(out["segments"][0], decode, duration)


def build_segments(raw_segments, decode, duration: float) -> List[Segment]:
    """
    Превращает сегменты из model.generate(return_segments=True) в список Segment.
    Таймкоды приводятся к монотонным и обрезаются по длительности: если модель не выдала
    стартовый таймкод, transformers возвращает мусорное (в том числе отрицательное) начало.
    """
    segments: List[Segment] = []
    prev_end = 0.0
    for seg in raw_segments:
        text = decode(seg["tokens"]).strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        if start < prev_end:
            start = prev_end
        start = min(start, duration)
        end = min(max(end, start), duration)
        segments.append(Segment(start, end, text))
        prev_end = end
    return segments


# -----------------------------------------------------------------------------
# ДИАРИЗАЦИЯ
# -----------------------------------------------------------------------------

def load_diarization_pipeline(model_name: str, device: str, hf_token: Optional[str]):
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    from pyannote.audio import Pipeline

    # Чекпоинты pyannote 3.x это pickle-файлы Lightning, а torch>=2.6 по умолчанию грузит
    # только веса. Разрешаем полную загрузку только на время создания пайплайна.
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        pipeline = Pipeline.from_pretrained(model_name)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous

    if pipeline is None:
        raise RuntimeError(
            f"Не удалось загрузить {model_name}. Нужен токен Hugging Face "
            f"(huggingface-cli login или --hf-token) и принятые условия модели на huggingface.co."
        )
    pipeline.to(torch.device(device))
    return pipeline


def diarize(audio: np.ndarray, model_name: str, device: str, hf_token: Optional[str],
            num_speakers: Optional[int], min_speakers: Optional[int], max_speakers: Optional[int]
            ) -> List[Tuple[float, float, str]]:
    """Возвращает отсортированный список реплик (start, end, speaker) от pyannote."""
    pipeline = load_diarization_pipeline(model_name, device, hf_token)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers

    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, N]
    result = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
    annotation = getattr(result, "speaker_diarization", result)  # pyannote 4.x оборачивает результат

    turns = [(float(turn.start), float(turn.end), str(speaker))
             for turn, _, speaker in annotation.itertracks(yield_label=True)]
    turns.sort()
    return turns


def assign_speakers(segments: List[Segment], turns: List[Tuple[float, float, str]]) -> None:
    """Каждому сегменту Whisper назначает спикера с наибольшим суммарным перекрытием по времени."""
    if not turns:
        return
    starts = np.array([t[0] for t in turns])
    ends = np.array([t[1] for t in turns])
    labels = [t[2] for t in turns]

    for seg in segments:
        overlap = np.minimum(ends, seg.end) - np.maximum(starts, seg.start)
        totals = {}
        for idx in np.nonzero(overlap > 0)[0]:
            totals[labels[idx]] = totals.get(labels[idx], 0.0) + float(overlap[idx])
        if totals:
            seg.speaker = max(totals, key=totals.get)
        else:
            # Сегмент не пересёкся ни с одной репликой: берём ближайшую по времени
            distance = np.maximum(starts - seg.end, seg.start - ends)
            seg.speaker = labels[int(np.argmin(distance))]


# -----------------------------------------------------------------------------
# СКЛЕЙКА И ВЫВОД
# -----------------------------------------------------------------------------

def merge_segments(segments: List[Segment], join_threshold: float, max_chars: int) -> List[Segment]:
    """Склеивает соседние сегменты одного спикера, если пауза между ними мала."""
    merged: List[Segment] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            if (prev.speaker == seg.speaker
                    and seg.start - prev.end <= join_threshold
                    and len(prev.text) + 1 + len(seg.text) <= max_chars):
                prev.text = f"{prev.text} {seg.text}"
                prev.end = max(prev.end, seg.end)
                continue
        merged.append(Segment(seg.start, seg.end, seg.text, seg.speaker))
    return merged


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def render(segments: List[Segment], with_timestamps: bool) -> str:
    lines = []
    for seg in segments:
        prefix = f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] " if with_timestamps else ""
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"{prefix}{speaker}{seg.text}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# ЗАПУСК
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Транскрибация видео/аудио в текст с таймкодами и спикерами. "
                    "Результат сохраняется рядом с исходным файлом.")
    p.add_argument("input", help="Путь к видео/аудио (mp4, mkv, mp3, wav, amr, m4a ...)")
    p.add_argument("-o", "--output", default=None,
                   help="Путь к txt-файлу (по умолчанию рядом с входным: <имя>_transcription.txt)")
    p.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"Код языка (ru, en ...). По умолчанию: {DEFAULT_LANGUAGE}")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Модель Whisper с Hugging Face. По умолчанию: {DEFAULT_MODEL}")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Устройство (по умолчанию auto)")
    # Диаризация
    p.add_argument("--no-diarize", action="store_true", help="Не размечать спикеров")
    p.add_argument("--diarization-model", default=DEFAULT_DIARIZATION_MODEL,
                   help=f"Модель pyannote. По умолчанию: {DEFAULT_DIARIZATION_MODEL}")
    p.add_argument("--hf-token", default=None, help="Токен Hugging Face (иначе берётся из huggingface-cli login / HF_TOKEN)")
    p.add_argument("--speakers", type=int, default=None, help="Точное число спикеров, если известно")
    p.add_argument("--min-speakers", type=int, default=None, help="Минимум спикеров")
    p.add_argument("--max-speakers", type=int, default=None, help="Максимум спикеров")
    # Формат вывода
    p.add_argument("--join-threshold", type=float, default=1.5,
                   help="Макс. пауза (с) для склейки соседних реплик одного спикера (по умолчанию 1.5)")
    p.add_argument("--max-line-chars", type=int, default=600, help="Лимит длины склеенной реплики (по умолчанию 600)")
    p.add_argument("--no-timestamps", action="store_true", help="Не выводить таймкоды")
    return p.parse_args()


def quiet_warnings() -> None:
    """Глушит известные безвредные предупреждения зависимостей (torchaudio, pyannote, lightning)."""
    for pattern in (
        r".*TorchAudio.*",                        # torchaudio 2.8 предупреждает о переходе на torchcodec
        r".*TensorFloat-32.*",                    # pyannote отключает TF32 ради воспроизводимости
        r".*degrees of freedom.*",                # pyannote pooling на очень коротких сегментах
        r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*",  # lightning сообщает о флаге, который мы ставим сами
    ):
        warnings.filterwarnings("ignore", message=pattern)


def main() -> None:
    args = parse_args()
    hf_logging.set_verbosity_error()
    quiet_warnings()
    torch.manual_seed(0)  # temperature fallback использует сэмплирование, фиксируем для воспроизводимости

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{stem}_transcription.txt")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    t0 = time.time()

    print(f"Декодирование аудио: {input_path}")
    audio = decode_audio(input_path, find_ffmpeg())
    duration = len(audio) / SAMPLE_RATE
    print(f"Длительность: {format_timestamp(duration)}")

    print(f"Загрузка Whisper ({args.model}) ...")
    processor, model = load_whisper(args.model, device)
    print("Транскрибация ...")
    segments = transcribe(audio, processor, model, args.language, device)
    print(f"Whisper выделил сегментов: {len(segments)}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    if not args.no_diarize:
        print(f"Диаризация ({args.diarization_model}) ...")
        try:
            turns = diarize(audio, args.diarization_model, device, args.hf_token,
                            args.speakers, args.min_speakers, args.max_speakers)
            assign_speakers(segments, turns)
            print(f"Спикеров найдено: {len({t[2] for t in turns})}")
        except Exception as e:
            print(f"ВНИМАНИЕ: диаризация не выполнена, результат будет без спикеров.\n  Причина: {e}")

    merged = merge_segments(segments, args.join_threshold, args.max_line_chars)
    text = render(merged, with_timestamps=not args.no_timestamps)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Реплик в результате: {len(merged)}")
    print(f"Сохранено: {output_path}")
    print(f"Готово за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
