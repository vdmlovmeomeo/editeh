import json
import re
from datetime import date
import ollama # pyright: ignore[reportMissingImports]
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

PROMPT = """
Đọc bảng bài tập viết tay trong ảnh. Chỉ trả về JSON array, không giải thích.
Mỗi dòng bài tập là một object: {"title": môn học, "content": việc cần làm, "deadline": "YYYY-MM-DD" hoặc null}.
Tách dạng "Môn: việc" thành title và content.
T2, T3, T4, T5, T6, T7, CN chỉ là thứ trong tuần, KHÔNG phải title và không được đưa vào deadline.
Nếu dòng có dạng "T5 (19/11): MẮC LỄ PHỤC" thì deadline là ngày 19/11 của năm gần nhất, content là "MẮC LỄ PHỤC".
Chuyển mọi ngày D/M hoặc DD/MM sang YYYY-MM-DD. Không dùng placeholder như ..., không tự bịa dữ liệu.
Chỉ dùng đúng các key title, content, deadline. Không viết chữ "vào title" hay "vào content" trong giá trị."""

MODEL = "qwen2.5vl:3b"
MAX_IMAGE_SIDE = 1280
IMAGE_QUALITY = 85


def _prepare_image(image: Path) -> Path:
    """Create a smaller temporary image so vision tokens fit in a 4096 context."""
    with Image.open(image) as source:
        source.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        prepared = source.convert("RGB")
        temporary = NamedTemporaryFile(suffix=".jpg", delete=False)
        temporary_path = Path(temporary.name)
        temporary.close()
        prepared.save(temporary_path, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
    return temporary_path
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
    prepared_image = _prepare_image(image)
    client = ollama.Client(host="http://localhost:11434", timeout=300)
    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT, "images": [str(prepared_image.resolve())]}],
            think=False,
            format="json",
            options={"temperature": 0, "num_predict": 256, "num_ctx": 4096},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Không gọi được Ollama: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        prepared_image.unlink(missing_ok=True)
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
    try:
        result = json.JSONDecoder().raw_decode(text[match.start() :])[0]
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AI tráº£ JSON khÃ´ng há»£p lá»‡: {text[:300]!r}"
        ) from exc
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        raise ValueError("AI không trả về task hợp lệ")
    # Fallback cho model nhỏ: tự tách dạng "math: workbook pg20".
    for item in result:
        if isinstance(item, dict):
            # Moondream thường dùng subject/workbook/date thay cho schema yêu cầu.
            if not item.get("title") and item.get("subject"):
                item["title"] = item.pop("subject")
            if not item.get("content") and item.get("workbook"):
                item["content"] = item.pop("workbook")
            if not item.get("deadline") and item.get("date"):
                item["deadline"] = item.pop("date")
            title = str(item.get("title", ""))
            if ":" in title and not str(item.get("content", "")).strip():
                subject, details = title.split(":", 1)
                item["title"] = subject.strip()
                item["content"] = details.strip()
            for key in ("title", "content", "deadline"):
                item.setdefault(key, None)
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
    image_path = Path(__file__).with_name("realsample.jpg")#ctx:4282tk
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(f"Lỗi: {exc}")
