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
from typing import List, Tuple, Optional

try:
    from faster_whisper import WhisperModel
except Exception as e:
    print("Не удалось импортировать faster_whisper. Убедитесь, что зависимости установлены.")
    raise

# --------------------------
# УТИЛИТЫ
# --------------------------

def find_ffmpeg() -> str:
    ffmpeg_name = "ffmpeg.exe" if platform.system().lower().startswith("win") else "ffmpeg"
    path_ffmpeg = shutil.which(ffmpeg_name)
    if path_ffmpeg:
        return path_ffmpeg
    base_dirs = []
    if getattr(sys, "_MEIPASS", None):
        base_dirs.append(os.path.join(sys._MEIPASS))
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
    raise FileNotFoundError("ffmpeg не найден. Добавьте его в PATH или положите в ./third_party/ffmpeg/bin/")

def extract_audio_to_wav16k_mono(input_media_path: str, ffmpeg_path: str) -> str:
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
    return " ".join(text.replace("\r", "\n").split()).strip()

def wrap_by_sentences(text: str) -> List[str]:
    if not text:
        return []
    out, buf = [], []
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
    if seconds < 0:
        seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# --------------------------
# ДИАРИЗАЦИЯ (опционально)
# --------------------------

def run_diarization(wav_path: str, hf_token: Optional[str], num_speakers: Optional[int]):
    from pyannote.audio import Pipeline
    kwargs = {}
    if num_speakers and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    if not hf_token:
        raise RuntimeError("Для диаризации нужен --hf-token (Hugging Face).")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline({"audio": wav_path}, **kwargs)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    segments.sort(key=lambda x: (x[0], x[1]))
    return segments

def assign_speakers_to_whisper_segments(whisper_segments: List[Tuple[float,float,str]],
                                        speaker_segments: List[Tuple[float,float,str]],
                                        default_label: str = "SPK00"):
    if not speaker_segments:
        return [(s,e,t,default_label) for (s,e,t) in whisper_segments]
    out = []
    i = 0
    n = len(speaker_segments)
    for (ws,we,wt) in whisper_segments:
        best_label, best_overlap = default_label, 0.0
        while i + 1 < n and speaker_segments[i+1][1] <= ws:
            i += 1
        for j in range(max(0, i-3), min(n, i+4)):
            ss,se,lab = speaker_segments[j]
            left, right = max(ws, ss), min(we, se)
            overlap = max(0.0, right - left)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, lab
        out.append((ws,we,wt,best_label))
    return out

def format_spk(label: str, prefix: str) -> str:
    digits = "".join(ch for ch in label if ch.isdigit())
    if digits == "":
        return f"{prefix}00"
    try:
        num = int(digits)
    except ValueError:
        return f"{prefix}00"
    return f"{prefix}{num:02d}"

# --------------------------
# СКЛЕЙКА
# --------------------------

def merge_segments_to_paragraphs(segments: List[Tuple[float,float,str]],
                                 join_threshold: float = 0.8,
                                 max_paragraph_chars: int = 1200) -> List[str]:
    if not segments:
        return []
    merged: List[str] = []
    cur_text = [segments[0][2]]
    prev_end = segments[0][1]

    def flush():
        nonlocal cur_text
        block = normalize_space(" ".join(cur_text))
        if not block:
            cur_text = []
            return
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

    for (s,e,t) in segments[1:]:
        gap = s - prev_end
        if gap <= join_threshold and len(" ".join(cur_text + [t])) <= max_paragraph_chars:
            cur_text.append(t)
        else:
            flush()
            cur_text = [t]
        prev_end = e
    flush()
    return merged

def merge_segments_by_speaker(labeled_segments: List[Tuple[float,float,str,str]],
                              speaker_prefix: str = "SPK",
                              join_threshold: float = 0.8,
                              max_paragraph_chars: int = 1200) -> List[str]:
    if not labeled_segments:
        return []
    out: List[str] = []
    cur_label = labeled_segments[0][3]
    cur_text = [labeled_segments[0][2]]
    prev_end = labeled_segments[0][1]

    def emit(label: str, buf: List[str]):
        if not buf:
            return
        text = normalize_space(" ".join(buf))
        if not text:
            return
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

    for (s,e,t,lab) in labeled_segments[1:]:
        gap = s - prev_end
        if lab == cur_label and gap <= join_threshold:
            if len(" ".join(cur_text + [t])) <= max_paragraph_chars:
                cur_text.append(t)
            else:
                emit(cur_label, cur_text)
                cur_text = [t]
        else:
            emit(cur_label, cur_text)
            cur_label = lab
            cur_text = [t]
        prev_end = e
    emit(cur_label, cur_text)
    return out

# --------------------------
# ОСНОВНОЙ PIPELINE
# --------------------------

def detect_default_compute(device: str) -> str:
    return "float16" if device == "cuda" else "int8"

def main():
    parser = argparse.ArgumentParser(description="Переносимая утилита транскрибации (faster-whisper + CUDA). TXT без таймкодов.")
    parser.add_argument("input", help="Путь к видео/аудио (mp4, mkv, mp3, wav, ...)")
    parser.add_argument("-o", "--output", default=None, help="Путь к результату (по умолчанию: рядом, .txt)")
    parser.add_argument("--model", default="large-v3", help="Модель faster-whisper: tiny/base/small/medium/large-v3")
    parser.add_argument("--model-dir", default=None, help="Локальная папка с моделью")
    parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto", help="Устройство")
    parser.add_argument("--compute-type", default=None, help="float16/float32/int8/int8_float16 ...")
    parser.add_argument("--language", default=None, help="Код языка (ru, en). Если не задан — авто.")
    parser.add_argument("--task", choices=["transcribe","translate"], default="transcribe", help="transcribe/translate")
    parser.add_argument("--vad", action="store_true", help="Включить VAD фильтр (лучшее разбиение)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument("--best-of", type=int, default=5, help="best_of (для sampling)")
    parser.add_argument("--join-threshold", type=float, default=0.8, help="Макс. пауза (сек) для склейки")
    parser.add_argument("--max-paragraph-chars", type=int, default=1200, help="Лимит длины абзаца")
    parser.add_argument("--quiet", action="store_true", help="Не печатать прогресс")
    # Диаризация
    parser.add_argument("--diarize", action="store_true", help="Разметка по спикерам (pyannote, нужен токен)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face токен для pyannote")
    parser.add_argument("--speakers", type=int, default=None, help="Подсказка числом спикеров")
    parser.add_argument("--speaker-prefix", default="SPK", help="Префикс меток спикеров")
    # Жёсткое требование GPU
    parser.add_argument("--gpu-required", action="store_true", help="Если CUDA недоступна/нет DLL — завершить с ошибкой (не откатываться на CPU)")
    args = parser.parse_args()

    ffmpeg_path = find_ffmpeg()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.abspath(args.output) if args.output else os.path.join(base_dir, f"{base_name}_transcript.txt")

    wav_path = extract_audio_to_wav16k_mono(input_path, ffmpeg_path)

    # Определяем устройство
    device = args.device
    if device == "auto":
        device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    compute_type = args.compute_type or detect_default_compute(device)

    model_source = args.model if not args.model_dir else os.path.abspath(args.model_dir)

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

    # Пытаемся загрузить GPU-модель, при неудаче — откат на CPU (если не запрещено)
    def make_model(dev: str, ctype: str):
        return WhisperModel(model_source, device=dev, compute_type=ctype)

    model = None
    tried_gpu = False
    if device == "cuda":
        tried_gpu = True
        try:
            model = make_model("cuda", compute_type)
        except Exception as e:
            msg = str(e)
            # характерные признаки отсутствия CUDA/cuDNN/cublas
            if any(k in msg.lower() for k in ["cudnn", "cublas", "cuda", "nvrtc", "could not locate", "dll", "cannot load"]):
                print("⚠️  CUDA недоступна или отсутствуют CUDA/cuDNN DLL. "
                      "Попробую переключиться на CPU (int8). "
                      "Чтобы требовать GPU — укажите --gpu-required.")
                if args.gpu-required:
                    raise
                device, compute_type = "cpu", "int8"
                model = make_model(device, compute_type)
            else:
                raise
    if model is None:
        model = make_model(device, compute_type)

    segments_iter, info = model.transcribe(
        wav_path,
        task=args.task,
        language=args.language,
        vad_filter=args.vad,
        beam_size=args.beam_size,
        best_of=args.best_of,
        word_timestamps=False
    )

    whisper_segments: List[Tuple[float,float,str]] = []
    idx = 0
    for seg in segments_iter:
        idx += 1
        text = seg.text.strip()
        start = float(seg.start); end = float(seg.end)
        whisper_segments.append((start, end, text))
        if not args.quiet:
            print(f"[{idx:04d}] {format_timestamp(start)} → {format_timestamp(end)} | {text}")

    # Диаризация (опционально)
    diar_ok = False
    diar_segments: List[Tuple[float,float,str]] = []
    if args.diarize:
        try:
            diar_segments = run_diarization(wav_path=wav_path, hf_token=args.hf_token, num_speakers=args.speakers)
            diar_ok = len(diar_segments) > 0
        except Exception as e:
            print(f"Диаризация недоступна: {e}")
            diar_ok = False

    if diar_ok:
        labeled = assign_speakers_to_whisper_segments(whisper_segments, diar_segments, default_label=f"{args.speaker_prefix}00")
        paragraphs = merge_segments_by_speaker(labeled, speaker_prefix=args.speaker_prefix,
                                               join_threshold=args.join_threshold,
                                               max_paragraph_chars=args.max_paragraph_chars)
    else:
        paragraphs = merge_segments_to_paragraphs(whisper_segments,
                                                  join_threshold=args.join_threshold,
                                                  max_paragraph_chars=args.max_paragraph_chars)

    final_text = "\n\n".join(paragraphs).strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"✅ TXT сохранён: {output_path}")
    print(f"⏱️  Готово за {time.time() - t0:.2f} c")

    try:
        os.remove(wav_path)
        os.rmdir(os.path.dirname(wav_path))
    except Exception:
        pass

if __name__ == "__main__":
    main()
