import os
import re
import logging
from typing import List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(
    r'(\[.*?\])|'  # Плейсхолдеры в []
    r'(\$.*?\$)|'  # Плейсхолдеры в $$
    r'(#.*?#!?)|'  # Цветные теги
    r'(@\w+!)|'  # Иконки
    r'({[^}]+})',  # Скрипты
    flags=re.DOTALL
)


def mask_placeholders(text: str) -> Tuple[str, List[str]]:
    """
    Находит все плейсхолдеры в тексте, заменяет их метками <PLH_0>, <PLH_1> и т.д.
    Возвращает:
      - результирующую строку
      - список найденных плейсхолдеров
    """
    placeholders = []

    def replacer(match):
        placeholders.append(match.group(0))
        return f"<PLH_{len(placeholders) - 1}>"

    masked_text = PLACEHOLDER_PATTERN.sub(replacer, text)
    return masked_text, placeholders


def unmask_placeholders(text: str, placeholders: List[str]) -> str:
    """
    Восстанавливает плейсхолдеры в строке из списка placeholders,
    заменяя <PLH_i> обратно на оригинальные конструкции.
    """
    for i, ph in enumerate(placeholders):
        text = text.replace(f"<PLH_{i}>", ph, 1)
    return text


def translate_via_api(
    text: str,
    src_lang: str,
    tgt_lang: str,
    client
) -> str:
    """
    Вызывает внешний API-механизм перевода, возвращает переведённый текст с сохранением всех плейсхолдеров.
    """
    try:
        # Маскируем плейсхолдеры
        masked_text, placeholders = mask_placeholders(text)
        print(masked_text)
        # Формируем чёткий контекст для перевода
        context_message = (
            f"You are translating a text from {src_lang} to {tgt_lang}. "
            "Do not translate or modify any placeholders or code blocks enclosed in brackets "
            "[...], dollar signs $...$, or curly braces {...}. "
            "Translate only the natural language text. Keep the placeholders exactly as they are."
        )

        # Вызываем API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": context_message},
                {"role": "user", "content": masked_text},
            ],
            stream=False,
            temperature=1.3,
        )

        # Извлекаем результат перевода
        translated_masked_text = response.choices[0].message.content

        # Восстанавливаем плейсхолдеры
        translated_text = unmask_placeholders(translated_masked_text, placeholders)

        return translated_text

    except Exception as e:
        logger.error(f"Ошибка при вызове API перевода: {str(e)}")
        return text

def create_client():
    return OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# -------------------------
# Пример списка тест-кейсов
# -------------------------
test_cases = [
    (
        "#high Very Rare#!",
        "english",
        "russian",
        "#high Очень Редко#!"
    ),
    (
        "\n\n#highlight You have a $trait_futa$ Scholar with you!#!",
        "english",
        "russian",
        "\n\n#highlight Vous avez un érudit $trait_futa$ avec vous!#!"
    ),
    # Другие тесты...
]


def run_tests(client):
    """
    Пример функции, которая прогоняет тесты, вызывая `translate_via_api`.
    """
    for i, (text, src, tgt, expected) in enumerate(test_cases, 1):
        try:
            # Проверяем корректность маскировки/восстановления (без реального перевода)
            masked, ph = mask_placeholders(text)
            restored = unmask_placeholders(masked, ph)
            assert restored == text, f"Test {i}: ошибка целостности плейсхолдеров"

            # Вызываем перевод (здесь фактически будет тот же текст, т.к. это пример)
            translated = translate_via_api(text, src, tgt, client)

            # Проверяем, что после перевода количество плейсхолдеров то же самое
            # (глубокая проверка может выглядеть иначе, но упрощённо проверим факты)
            placeholders_in_translated = PLACEHOLDER_PATTERN.findall(translated)
            assert len(placeholders_in_translated) == len(ph), f"Test {i}: плейсхолдеры не совпадают {placeholders_in_translated} vs {ph} {translated}"

            print(f"Test {i} passed. Original vs Translated:\n - {text}\n - {translated}\n")

        except AssertionError as e:
            print(f"Test {i} failed: {str(e)}")


# -------------------------
# Пример использования
# -------------------------
if __name__ == "__main__":
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    # run_tests(client)
    # Или просто единоразово вызвать:
    #
    result = translate_via_api("It is almost time. My plan to claim [target.GetFirstNamePossessive] heart #italic and#! body has almost come to fruition. There is only a single question left on my mind. Should I make this a regular thing with [target.GetFirstName], or have this be a one time only event?", "englisn", "russian", client)
    print(result)
