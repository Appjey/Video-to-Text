import torch
import os
import torchaudio
import moviepy.editor as mp

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Пути для видео, аудио и файла с результатами транскрипции
filename = "2026-01-28 13-01-29"
video_path = f"./mnt/data/{filename}.mp4"
audio_path = f"./mnt/data/{filename}.mp4"
language = "ru"
transcription_file_path = f"./mnt/data/{os.path.splitext(os.path.basename(audio_path))[0]}_transcription.txt"
wav_path = ""

# Максимальная длина фрагмента в секундах
MAX_AUDIO_LENGTH_SEC = 30

# -----------------------------------------------------------------------------
# ФУНКЦИИ
# -----------------------------------------------------------------------------

def extract_audio_from_video(video_path, audio_path):
    """Извлекает аудио из видеофайла с помощью MoviePy."""
    if os.path.exists(audio_path):
        print(f"Аудио уже извлечено: {audio_path}")
        return
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    print(f"Аудио сохранено в {audio_path}")

def convert_to_wav_using_moviepy(input_path, output_wav_path):
    """
    Конвертирует входной аудиофайл (например, M4A) в WAV с помощью MoviePy (ffmpeg).
    codec='pcm_s16le' даёт 16-битный PCM, что удобно для дальнейшей обработки.
    """
    if os.path.exists(output_wav_path):
        print(f"WAV файл уже существует: {output_wav_path}")
        return
    print(f"Конвертация файла {input_path} в WAV...")
    # Открываем аудио через MoviePy
    with mp.AudioFileClip(input_path) as audio_clip:
        # Сохраняем в WAV (16-бит PCM)
        audio_clip.write_audiofile(output_wav_path, codec='pcm_s16le')
    print(f"Файл сохранён как WAV: {output_wav_path}")

def load_and_resample_audio(audio_path, target_sample_rate=16000):
    """
    Загружает аудио (через torchaudio), ресемплирует до 16000 Гц и сводит к моно при необходимости.
    Возвращает тензор с формой [1, сигналы], а также частоту дискретизации.
    """
    speech_array, original_sample_rate = torchaudio.load(audio_path)

    # Если аудио стерео, приводим в моно
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    # Приводим частоту дискретизации к target_sample_rate (по умолчанию 16 кГц)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate,
            new_freq=target_sample_rate
        )
        speech_array = resampler(speech_array)

    return speech_array, target_sample_rate

def check_and_resample_audio(audio_path, desired_sample_rate=16000, desired_channels=1):
    """
    Проверяет расширение и параметры аудиофайла. Если не WAV — конвертирует в WAV.
    Затем проверяет, нужна ли ресэмплинг/приведение в моно. Возвращает путь к конечному (WAV) файлу.
    """
    base, ext = os.path.splitext(audio_path)
    global wav_path
    # Определяем, нужно ли делать принудительную конвертацию в WAV
    if ext.lower() != ".wav":
        wav_path = f"{base}_converted.wav"
        convert_to_wav_using_moviepy(audio_path, wav_path)
        # Далее работаем уже с wav-путём
        final_path = wav_path
    else:
        final_path = audio_path

    # Теперь проверяем частоту дискретизации и канальность
    #  (загружаем в торч; если не подходит — пересохраняем в нужном формате)
    speech_array, original_sample_rate = torchaudio.load(final_path)
    current_channels = speech_array.shape[0]

    if (original_sample_rate != desired_sample_rate) or (current_channels != desired_channels):
        print("Частота дискретизации/каналы не соответствуют, пересохраняем...")
        # Приводим к моно
        if current_channels > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        # Приводим к нужной частоте
        if original_sample_rate != desired_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=desired_sample_rate
            )
            speech_array = resampler(speech_array)

        # Сохраняем окончательно как WAV
        final_wav_path = f"{base}_final.wav"
        torchaudio.save(final_wav_path, speech_array, desired_sample_rate)
        final_path = final_wav_path

    return final_path

def split_audio_into_chunks(speech_array, sample_rate, chunk_length_sec=30):
    """
    Разбивает аудио на чанки (в секундах). Возвращает список тензоров.
    """
    chunk_length = chunk_length_sec * sample_rate
    num_chunks = (speech_array.size(1) + chunk_length - 1) // chunk_length
    chunks = [speech_array[:, i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
    return chunks

def transcribe_chunk_with_whisper(chunk, model, processor, sample_rate, device):
    chunk = chunk.squeeze().numpy()
    inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda"):  # dtype по умолчанию float16
                predicted_ids = model.generate(inputs, language=language, task="transcribe")
        else:
            predicted_ids = model.generate(inputs, language=language, task="transcribe")

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def transcribe_with_whisper_in_chunks(audio_path, chunk_length_sec=30):
    """
    Основная функция транскрибации: загружает и разбивает аудио, транскрибирует чанки,
    собирает и возвращает итоговый текст.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")

    # Загружаем модель и процессор
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)
    model.config.forced_decoder_ids = None  # отключаем forced decoder ids

    # Загружаем аудио, ресэмплим, приводим в моно
    speech_array, sample_rate = load_and_resample_audio(audio_path)

    # Разбиваем на фрагменты
    chunks = split_audio_into_chunks(speech_array, sample_rate, chunk_length_sec)

    # Транскрибируем все чанки
    transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Транскрибируется фрагмент {i+1}/{len(chunks)} ...")
        chunk_transcription = transcribe_chunk_with_whisper(chunk, model, processor, sample_rate, device)
        transcription += chunk_transcription + " "

    return transcription.strip()

def process_video_to_transcription(video_path, audio_path, transcription_file_path):
    """
    Финальная функция: (опционально) извлекает аудио из видео,
    приводит аудио к корректному формату,
    транскрибирует чанками через Whisper,
    сохраняет результат в txt.
    """
    # 1. Извлечение аудио из видео (при необходимости раскомментируйте):
    # extract_audio_from_video(video_path, audio_path)

    # 2. Проверка и приведение аудио к нужным параметрам
    checked_audio_path = check_and_resample_audio(audio_path)

    # 3. Запускаем транскрибацию на корректном WAV
    transcription = transcribe_with_whisper_in_chunks(checked_audio_path)

    # 4. Сохраняем результат
    with open(transcription_file_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    print(f"Транскрипция завершена и сохранена в {transcription_file_path}")

    os.remove(checked_audio_path)
    os.remove(wav_path)


# -----------------------------------------------------------------------------
# ЗАПУСК ПРОЦЕССА
# -----------------------------------------------------------------------------
process_video_to_transcription(video_path, audio_path, transcription_file_path)
