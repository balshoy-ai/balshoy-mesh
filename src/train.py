import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 1. Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Конфигурация 4-битного квантования (QLoRA) для слабого железа
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 3. Загрузка базовой модели с квантованием
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Подготовка модели к обучению в пониженной точности
    model = prepare_model_for_kbit_training(model)

    # 4. Настройка LoRA-адаптеров (обучаем только ~1% весов)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # основные слои внимания Qwen
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Подключение датасета в режиме СТРИМИНГА
    # Загружаем как 'json', указав путь к нашему JSONL-файлу
    dataset = load_dataset("json", data_files="data/dataset.jsonl", streaming=True)
    
    # Функция форматирования текста под шаблон модели Qwen
    def tokenize_function(examples):
        text = f"User: {examples['instruction']}\nAssistant: {examples['output']}"
        outputs = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # Применяем токенизацию к потоку данных
    tokenized_dataset = dataset["train"].map(tokenize_function, batched=False)

    # 6. Параметры обучения (оптимизировано под экономию памяти)
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        per_device_train_batch_size=1,        # минимальный батч для экономии VRAM
        gradient_accumulation_steps=4,       # компенсируем маленький батч накоплением градиентов
        max_steps=100,                        # для стриминга используем max_steps вместо epochs
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        optim="paged_adamw_8bit",            # 8-битный оптимизатор экономит память
        remove_unused_columns=False,
        gradient_checkpointing=True           # критично для слабого железа
    )

    # 7. Запуск обучения
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("Запуск процесса дообучения...")
    trainer.train()
    
    # 8. Сохранение обученного LoRA-адаптера
    model.save_pretrained("models/lora_adapter")
    print("Обучение завершено. Адаптер сохранен в models/lora_adapter")

if __name__ == "__main__":
    main()
