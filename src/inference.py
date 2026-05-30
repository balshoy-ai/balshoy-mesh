"""Локальная проверка модели (инференс)."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model():
    base_model_name = "models/base"
    adapter_path = "models/checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def generate(prompt: str, max_length: int = 128):
    model, tokenizer = load_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(generate("Привет, как дела?"))
