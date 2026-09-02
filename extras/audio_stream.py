import pyaudio
import numpy as np
import torch
import threading
import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Инициализация модели и процессора Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)

# Установка параметров записи
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper ожидает аудио с частотой дискретизации 16kHz
CHUNK = 1024  # Размер фрейма для обработки
BUFFER_DURATION = 5  # Длительность буфера в секундах

def transcribe_stream():
    # Инициализация PyAudio
    audio = pyaudio.PyAudio()

    # Открытие аудиопотока
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Начало обработки аудиопотока...")

    frames = []

    try:
        while True:
            # Чтение аудиоданных
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))

            # Если накоплено достаточно данных для обработки
            if len(frames) * CHUNK >= RATE * BUFFER_DURATION:
                # Объединение всех фреймов в один массив
                audio_data = np.hstack(frames)

                # Преобразование аудио в формат float32
                audio_data = audio_data.astype(np.float32) / 32768.0

                # Преобразование аудиоданных в правильный формат для Whisper
                inputs = processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_features.to(device)

                # Генерация текста с помощью модели Whisper
                predicted_ids = model.generate(inputs)

                # Декодирование текста
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                # Вывод распознанного текста
                print("Распознанный текст:", transcription)

                # Очистка фреймов
                frames = []

    except KeyboardInterrupt:
        # Остановка потока
        print("Обработка завершена")
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Запуск функции
transcribe_stream()