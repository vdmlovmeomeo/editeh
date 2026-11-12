import json
from datetime import date
import ollama

PROMPT = """Đọc ảnh này và liệt kê các công việc/deadline có trong đó.
Chỉ trả về JSON hợp lệ, không giải thích, không dùng markdown.
Định dạng: [{"title": "...", "content": "...", "deadline": "YYYY-MM-DD"}]
Nếu không có deadline rõ ràng thì bỏ qua mục đó."""

def extract_tasks(image_path: str) -> list[dict]:
    response = ollama.chat(
        model="qwen3vl:2b",
        messages=[{"role": "user", "content": PROMPT, "images": [image_path]}],
        thinking=False,
        )
    text = response["message"]["content"].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)

def import_from_image(manager, image_path: str) -> int:
    added = 0
    for item in extract_tasks(image_path):
        try:
            t = str(item["title"]).strip()
            c = str(item.get("content", ""))
            d = date.fromisoformat(item["deadline"])
            manager.add_task(t, c, {"dd": d.day, "mm": d.month, "yy": d.year})
            added += 1
        except (KeyError, ValueError, TypeError):
            ...

    manager.save_to_csv()
    return added