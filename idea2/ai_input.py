import json
import re
from datetime import date
import ollama
from pathlib import Path

PROMPT = """/no_think
Đọc ảnh này và liệt kê các công việc/deadline có trong đó.
Chỉ trả về JSON hợp lệ, không giải thích, không dùng markdown.
Định dạng: [{"title": "...", "content": "...", "deadline": "YYYY-MM-DD"}]
Quy tắc tách nội dung:
- Nếu dòng có dạng "Môn/chủ đề: việc cần làm", đặt phần trước dấu ":" vào title.
- Đặt phần sau dấu ":" vào content.
- Ví dụ "math: workbook pg20" phải trả về title="math", content="workbook pg20".
- Không tự thêm thông tin không có trong ảnh.
Nếu không có deadline rõ ràng thì bỏ qua mục đó."""

MODEL = "qwen3.5:0.8b"

def parse_deadline(value: str) -> date:
    """Parse ISO dates and short dates such as 1/1 or 01/01."""
    value = str(value).strip()
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass

    parts = value.replace("-", "/").split("/")
    if len(parts) == 2:
        first, second = (int(part) for part in parts)
        # Model thường trả M/D; nếu phần đầu > 12 thì hiểu là D/M.
        month, day = (first, second) if first <= 12 else (second, first)
        today = date.today()
        year = today.year
        candidate = date(year, month, day)
        if candidate < today:
            candidate = date(year + 1, month, day)
        return candidate
    raise ValueError(f"Deadline không hợp lệ: {value!r}")

def extract_tasks(image_path: str) -> list[dict]:
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image}")

    # Ollama chat là blocking. Timeout giúp không bị treo vô hạn khi Ollama
    # chưa chạy hoặc model chưa được cài.
    # Dùng localhost để phù hợp cả cấu hình Ollama bind theo hostname.
    # Vision model có thể mất vài phút khi chạy CPU hoặc load lần đầu.
    client = ollama.Client(host="http://localhost:11434", timeout=300)
    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT, "images": [str(image.resolve())]}],
            think=False,
            format="json",
            options={"temperature": 0, "num_predict": 1024, "num_ctx": 2048},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Không gọi được Ollama: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    message = response.get("message", {})
    text = (message.get("content") or "").strip()
    if not text:
        text = (message.get("thinking") or "").strip()
    # Không để phần suy nghĩ của Qwen làm hỏng JSON.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Model đôi khi thêm lời giải thích/markdown trước JSON.
    match = re.search(r"(?:\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"AI không trả về JSON danh sách task. Response: {text[:300]!r}")
    result = json.loads(match.group(0))
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        raise ValueError("AI không trả về task hợp lệ")
    # Fallback cho model nhỏ: tự tách dạng "math: workbook pg20".
    for item in result:
        if isinstance(item, dict):
            title = str(item.get("title", ""))
            if ":" in title and not str(item.get("content", "")).strip():
                subject, details = title.split(":", 1)
                item["title"] = subject.strip()
                item["content"] = details.strip()
    return result

def import_from_image(manager, image_path: str) -> int:
    added = 0
    for item in extract_tasks(image_path):
        try:
            t = str(item["title"]).strip()
            c = str(item.get("content", ""))
            d = parse_deadline(item["deadline"])
            manager.add_task(t, c, {"dd": d.day, "mm": d.month, "yy": d.year})
            added += 1
        except (KeyError, ValueError, TypeError):
            ...

    manager.save_to_csv()
    return added

if __name__ == "__main__":
    image_path = Path(__file__).with_name("testimg.png")
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(f"Lỗi: {exc}")
