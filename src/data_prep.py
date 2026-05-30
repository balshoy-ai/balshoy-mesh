import json
import os

def load_raw_txt(file_path):
    """Пример чтения обычного текстового файла (например, инструкции)."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "Инструкция по умолчанию: всегда перезагружайте сервер при ошибке."

def load_mock_sql_data():
    """Имитация выгрузки данных из SQL-базы и перевод их в текст."""
    # Вместо этого блока в реальном проекте будет: cursor.execute("SELECT...")
    mock_db_rows = [
        {"id": 101, "username": "Aleksey", "status": "active", "balance": 150},
        {"id": 102, "username": "Elena", "status": "banned", "balance": 0}
    ]
    
    textualized_data = []
    for row in mock_db_rows:
        text_line = f"Пользователь {row['username']} (ID: {row['id']}) имеет статус {row['status']}. Баланс: {row['balance']} рублей."
        textualized_data.append(text_line)
    return textualized_data

def main():
    output_path = "data/dataset.jsonl"
    os.makedirs("data", exist_ok=True)
    
    # 1. Собираем данные из разных источников
    txt_content = load_raw_txt("data/raw/instruction.txt")
    sql_lines = load_mock_sql_data()
    
    # 2. Формируем пары "Инструкция - Ответ" для обучения (Alpaca-формат)
    dataset_entries = [
        {
            "instruction": "Каковы базовые правила администрирования?",
            "output": txt_content
        }
    ]
    
    for line in sql_lines:
        dataset_entries.append({
            "instruction": "Предоставь информацию о пользователе из базы данных.",
            "output": line
        })
        
    # 3. Записываем в формате JSON Lines (каждая строка — отдельный JSON)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Успешно создано {len(dataset_entries)} записей в {output_path}")

if __name__ == "__main__":
    main()
