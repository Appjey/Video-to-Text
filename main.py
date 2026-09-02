#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Транскрибация видео/аудио в текст через Whisper (transformers).

Примеры:
    python main.py "mnt/data/запись.mp4"
    python main.py "mnt/data/звонок.amr" --language ru --output результат.txt

Результат сохраняется рядом с исходным файлом как <имя файла>_transcription.txt.
Аудио извлекается через ffmpeg (из PATH или из пакета imageio-ffmpeg).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_LANGUAGE = "ru"
TARGET_SAMPLE_RATE = 16000   # Whisper ожидает 16 кГц, моно
CHUNK_LENGTH_SEC = 30        # Whisper принимает фрагменты не длиннее 30 секунд


# -----------------------------------------------------------------------------
# ПОДГОТОВКА АУДИО
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
        raise FileNotFoundError(
            "ffmpeg не найден. Добавьте его в PATH или установите пакет imageio-ffmpeg."
        )


def convert_to_wav16k_mono(input_path: str, output_wav_path: str, ffmpeg_path: str) -> None:
    """
    Извлекает звуковую дорожку из любого видео/аудио (mp4, mkv, m4a, amr, mp3, wav ...)
    и сохраняет её как WAV 16 кГц, моно, 16-бит PCM.
    """
    print(f"Извлечение аудио из {input_path} ...")
    cmd = [
        ffmpeg_path, "-y", "-loglevel", "error",
        "-i", input_path,
        "-vn",                          # без видео
        "-ac", "1",                     # моно
        "-ar", str(TARGET_SAMPLE_RATE), # 16 кГц
        "-c:a", "pcm_s16le",            # 16-бит PCM
        "-fflags", "+bitexact",         # без служебных чанков в заголовке
        "-f", "wav",
        output_wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not os.path.exists(output_wav_path):
        raise RuntimeError(f"ffmpeg не смог извлечь аудио:\n{proc.stderr}")


def load_wav_as_float32(wav_path: str) -> np.ndarray:
    """Читает WAV 16 кГц/моно/16-бит и возвращает массив float32 в диапазоне [-1, 1]."""
    with wave.open(wav_path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != TARGET_SAMPLE_RATE or w.getsampwidth() != 2:
            raise ValueError(
                f"Ожидался WAV {TARGET_SAMPLE_RATE} Гц/моно/16-бит, получено: "
                f"{w.getframerate()} Гц, каналов {w.getnchannels()}, {w.getsampwidth() * 8}-бит"
            )
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def split_audio_into_chunks(audio: np.ndarray, sample_rate: int, chunk_length_sec: int = CHUNK_LENGTH_SEC):
    """Разбивает аудио на фрагменты по chunk_length_sec секунд. Возвращает список массивов."""
    chunk_length = chunk_length_sec * sample_rate
    return [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]


# -----------------------------------------------------------------------------
# ТРАНСКРИБАЦИЯ
# -----------------------------------------------------------------------------

def transcribe_chunk(chunk: np.ndarray, model, processor, sample_rate: int, device: str, language: str) -> str:
    inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                predicted_ids = model.generate(inputs, language=language, task="transcribe")
        else:
            predicted_ids = model.generate(inputs, language=language, task="transcribe")

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def transcribe_audio(wav_path: str, model_name: str, language: str) -> str:
    """Загружает модель и аудио, транскрибирует по фрагментам и возвращает итоговый текст."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.config.forced_decoder_ids = None

    audio = load_wav_as_float32(wav_path)
    print(f"Длительность аудио: {len(audio) / TARGET_SAMPLE_RATE:.1f} с")
    chunks = split_audio_into_chunks(audio, TARGET_SAMPLE_RATE)

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"Транскрибируется фрагмент {i}/{len(chunks)} ...")
        parts.append(transcribe_chunk(chunk, model, processor, TARGET_SAMPLE_RATE, device, language))

    return " ".join(part.strip() for part in parts).strip()


def process_media_to_transcription(input_path: str, output_path: str, model_name: str, language: str) -> None:
    """
    Полный цикл: извлекает аудио во временный WAV, транскрибирует и сохраняет
    текст в output_path. Временные файлы удаляются, исходный файл не трогается.
    """
    ffmpeg_path = find_ffmpeg()
    tmp_dir = tempfile.mkdtemp(prefix="vide_to_text_")
    try:
        wav_path = os.path.join(tmp_dir, "audio_16k_mono.wav")
        convert_to_wav16k_mono(input_path, wav_path, ffmpeg_path)

        transcription = transcribe_audio(wav_path, model_name, language)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"Транскрипция завершена и сохранена в {output_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# ЗАПУСК
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Транскрибация видео/аудио в текст (Whisper). Результат сохраняется рядом с исходным файлом."
    )
    parser.add_argument("input", help="Путь к видео/аудио (mp4, mkv, mp3, wav, amr, m4a ...)")
    parser.add_argument("-o", "--output", default=None,
                        help="Путь к txt-файлу с результатом (по умолчанию рядом с входным: <имя>_transcription.txt)")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE,
                        help=f"Код языка (ru, en ...). По умолчанию: {DEFAULT_LANGUAGE}")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Модель Whisper с Hugging Face. По умолчанию: {DEFAULT_MODEL}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{stem}_transcription.txt")

    process_media_to_transcription(input_path, output_path, args.model, args.language)


if __name__ == "__main__":
    main()
