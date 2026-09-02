from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Определение модели и токенизатора
model_id = "nvidia/Llama3-ChatQA-2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Чтение текста из файла
file_path = "./mnt/data/Перевод статусов релиза_transcription.txt"
with open(file_path, "r", encoding="utf-8") as f:
    document = f.read()

# Вопрос пользователя
messages = [
    {"role": "user", "content": "what are the key agreements and actions from the meeting?"}
]


def get_formatted_input(messages, context):
    system = ("System: This is a business meeting transcription analysis assistant. "
              "The assistant provides structured, clear, and complete summaries of key agreements, action items, deadlines, and responsibilities from the provided transcription. "
              "If specific details are not found in the context, the assistant should indicate their absence and suggest potential next steps.")
    instruction = ("Please analyze the transcript carefully and provide a thorough and detailed summary, including: \n"
                   "1. Key agreements. \n"
                   "2. Actionable tasks. \n"
                   "3. Assigned responsibilities. \n"
                   "4. Deadlines. \n"
                   "Ensure the response is structured and comprehensive.")

    for item in messages:
        if item['role'] == "user":
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(
        ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in
         messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation

    return formatted_input


formatted_input = get_formatted_input(messages, document)
tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Генерация ответа модели
outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask,
                         max_new_tokens=512, eos_token_id=terminators)

response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
