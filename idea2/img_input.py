import json
import re
from datetime import date
import ollama # pyright: ignore[reportMissingImports]
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

PROMPT = """
Ảnh là bảng bài tập viết tay. Mỗi dòng bài tập là một task riêng. Đọc từ trên xuống dưới, không bỏ dòng nào.

Trả về JSON dạng {"tasks": [...]}. Mỗi phần tử có đúng 3 key: "title", "content", "deadline".

Quy tắc:
1. title: chỉ là tên môn học, ví dụ "ESL", "Sử", "Toán", "Văn", "Lý", "Anh". Không đưa dấu ":" hay nội dung việc vào title.
2. content: việc cần làm, không kèm tên môn. Ví dụ "ESL: Flyer" thì title "ESL", content "Flyer". "Toán: Lý thuyết : 5 chương" thì title "Toán", content "Lý thuyết : 5 chương".
3. Dòng không ghi môn thì lấy môn của dòng môn gần nhất phía trên. Ví dụ "Workbook p42" nằm dưới "ESL" thì title là "ESL".
4. Dòng chỉ là tiêu đề môn như "Toán:" không phải task, không xuất ra. Dùng môn đó cho các dòng bên dưới.
5. deadline: ngày hoặc thứ ở cột bên phải cùng dòng, viết dạng YYYY-MM-DD. D/M là ngày/tháng. Dùng năm hiện tại; nếu ngày đó đã qua thì dùng năm kế tiếp. Thứ T2..T7, CN đổi sang ngày gần nhất theo mốc thời gian bên dưới. Nếu dòng không có ngày hoặc thứ thì để null.
6. Tiêu đề bảng như "Date", "HW", "Deadline" không phải task, bỏ qua.
7. Không tự bịa chữ khó đọc, giữ phần đọc được.
8. Một môn có nhiều việc con (nhiều dòng hoặc có nhánh) thì tách thành nhiều task, mỗi việc con là một task riêng, cùng title và deadline. Ví dụ ESL có 3 việc thì ra 3 task.
9. Tên môn viết chuẩn: Toán, Văn, Anh, ESL, Sử, Địa, Lý, Hóa, Sinh học, Tin. Chữ viết tay giống "Uẩn" hoặc "Uần" thì là "Văn".
10. Chỉ trả JSON, không giải thích.
"""
"""
minicpm-v
minicpm-v4.6
lfm3-omni:2b
lfm3
lfm3-vl
gemma3:4b
gemma3n:e2b
gemma3n:e4b
nemotron-3.5-bolt:e4b
bonsai-2:2b
"""

MODEL = "lfm3-omni:2b"
# MODEL = 'gemma3n:e2b'
MAX_IMAGE_SIDE = 1024
IMAGE_QUALITY = 100
STRIP_OVERLAP = 50


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

def parse_deadline(value) -> date:
    """Lấy ngày/tháng từ chuỗi và chọn năm gần hôm nay nhất (năm trước, năm nay hoặc năm sau).
    Chấp nhận ISO do code tạo, DD/MM do model trả, và YYYY-DD-MM do model hay trả sai."""
    text = str(value).strip()
    today = date.today()

    try:
        parsed = date.fromisoformat(text)
        if abs(parsed.year - today.year) <= 1:
            return parsed
    except ValueError:
        pass

    parts = [p for p in re.split(r"[/.-]", text) if p.isdigit()]
    if len(parts) >= 3 and len(parts[0]) == 4:
        # YYYY-DD-MM: bỏ năm, hai số sau là ngày và tháng theo thứ tự model hay trả.
        day, month = int(parts[1]), int(parts[2])
    elif len(parts) >= 2:
        # DD/MM hoặc M/D: nếu số sau > 12 thì không thể là tháng, đảo lại.
        first, second = int(parts[0]), int(parts[1])
        day, month = (first, second) if second <= 12 else (second, first)
    else:
        raise ValueError(f"Deadline không hợp lệ: {text!r}")

    candidates = []
    for year in (today.year - 1, today.year, today.year + 1):
        try:
            candidates.append(date(year, month, day))
        except ValueError:
            continue
    if not candidates:
        raise ValueError(f"Deadline không hợp lệ: {text!r}")
    return min(candidates, key=lambda d: abs((d - today).days))


def _weekday_deadline(value: str, reference: date | None = None) -> date | None:
    """Convert Vietnamese weekday notes (T2..T7/CN) to the next calendar date."""
    text = str(value or "").strip().casefold()
    match = re.search(r"(?<!\w)(t([2-7])|cn)(?:\s*\(([^)]*tuần[^)]*)\))?", text)
    if not match:
        return None
    token, number, qualifier = match.groups()
    target = 6 if token == "cn" else int(number) - 2  # Monday=0
    today = reference or date.today()
    days_ahead = (target - today.weekday()) % 7
    # A bare weekday means the next occurrence; same-day means today.
    if days_ahead == 0 and today.weekday() != target:
        days_ahead = 7
    if qualifier and "sau" in qualifier:
        days_ahead += 7
    return today.fromordinal(today.toordinal() + days_ahead)


def _normalize_task(item: dict) -> dict:
    """Recover explicit day/month dates the vision model put in another field."""
    for key in ("title", "content"):
        value = str(item.get(key) or "")
        match = re.search(
            r"(?:(T[2-7]|CN)\s*)?\(?\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{4}))?\)?",
            value,
            re.IGNORECASE,
        )
        if not match:
            continue
        weekday_text, day_text, month_text, year_text = match.groups()
        day, month = int(day_text), int(month_text)
        if year_text:
            candidate = date(int(year_text), month, day)
        elif weekday_text:
            # A weekday plus day/month identifies the intended school year. Pick
            # the matching weekday closest to today, allowing a date just passed.
            weekday_number = 6 if weekday_text.upper() == "CN" else int(weekday_text[1]) - 2
            candidates = [
                date(year, month, day)
                for year in (date.today().year - 1, date.today().year, date.today().year + 1)
                if date(year, month, day).weekday() == weekday_number
            ]
            candidate = min(candidates, key=lambda value: abs((value - date.today()).days)) if candidates else date(date.today().year, month, day)
        else:
            candidate = date(date.today().year, month, day)
            if candidate < date.today():
                candidate = date(candidate.year + 1, month, day)
        if not item.get("deadline"):
            item["deadline"] = candidate.isoformat()
        value = value[:match.start()] + value[match.end():]
        item[key] = re.sub(r"\s+", " ", value).strip(" \t:;-()")

    # Weekday notes are often returned in deadline, or appended to the right
    # of the task text. Resolve them before validating explicit date strings.
        # Thứ (T2..CN) có thể nằm ở deadline hoặc cuối title/content.
    deadline_text = str(item.get("deadline") or "")
    if deadline_text and not re.search(r"\d", deadline_text):
        weekday = _weekday_deadline(deadline_text)
        item["deadline"] = weekday.isoformat() if weekday else None
    for key in ("title", "content"):
        value = str(item.get(key) or "")
        weekday = _weekday_deadline(value)
        if weekday:
            if not item.get("deadline"):
                item["deadline"] = weekday.isoformat()
            item[key] = re.sub(
                r"(?<!\w)(?:T[2-7]|CN)(?:\s*\([^)]*tuần[^)]*\))?",
                "",
                value,
                flags=re.IGNORECASE,
            )
            item[key] = re.sub(r"\s+", " ", item[key]).strip(" \t:;-()")
            
    if item.get("deadline"):
        try:
            parsed_deadline = parse_deadline(str(item["deadline"]))
            # Old years are usually vision hallucinations (the board shows
            # the current school year). Do not expose a stale deadline.
            if parsed_deadline.year < date.today().year:
                item["deadline"] = None
            else:
                item["deadline"] = parsed_deadline.isoformat()
        except ValueError:
            pass
    return item


def extract_tasks(image_path: str) -> list[dict]:
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Kh\u00f4ng t\u00ecm th\u1ea5y \u1ea3nh: {image}")

    # Send the original board as one image; do not split it into strips.
    prepared_images = [_prepare_image(image)]
    today = date.today()
    weekday_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    date_context = (
        f"\nM\u1ed1c th\u1eddi gian: h\u00f4m nay l\u00e0 {weekday_names[today.weekday()]} "
        f"ng\u00e0y {today.day:02d}/{today.month:02d}/{today.year}."
        "\nTr\u00edch xu\u1ea5t c\u00e1c task trong \u1ea3nh v\u00e0 tr\u1ea3 v\u1ec1 JSON theo quy t\u1eafc tr\u00ean."
    )
    client = ollama.Client(host="http://localhost:11434", timeout=300)
    tasks: list[dict] = []
    try:
        for prepared_image in prepared_images:
            response = client.chat(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": PROMPT + date_context,
                    "images": [str(prepared_image.resolve())],
                }],
                format="json",
                options={"temperature": 0, "num_predict": 2048, "num_ctx": 8192, "num_gpu": 0},
                stream=False,
            )
            text = (response.get("message", {}).get("content") or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"(?:\[.*\]|\{.*\})", text, flags=re.DOTALL)
                if not match:
                    continue
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
            if isinstance(parsed, dict):
                # Vision models often wrap the array in a named property.
                wrapped = next(
                    (parsed[key] for key in ("tasks", "entries", "items", "data", "results")
                     if isinstance(parsed.get(key), list)),
                    None,
                )
                parsed = wrapped if wrapped is not None else [parsed]
            if isinstance(parsed, list):
                tasks.extend(item for item in parsed if isinstance(item, dict))
    except Exception as exc:
        raise RuntimeError(f"Kh\u00f4ng g\u1ecdi \u0111\u01b0\u1ee3c Ollama: {type(exc).__name__}: {exc}") from exc
    finally:
        for prepared_image in prepared_images:
            prepared_image.unlink(missing_ok=True)

    result: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in tasks:
        item = dict(item)
        if not item.get("title") and item.get("subject"):
            item["title"] = item.pop("subject")
        if not item.get("content") and item.get("workbook"):
            item["content"] = item.pop("workbook")
        if not item.get("deadline") and item.get("date"):
            item["deadline"] = item.pop("date")
        # Model đôi khi gộp "Môn : việc" vào title; tách lại khi content trống.
        title = str(item.get("title") or "")
        if ":" in title and not str(item.get("content") or "").strip():
            head, tail = title.split(":", 1)
            item["title"], item["content"] = head.strip(), tail.strip()
        for key in ("title", "content", "deadline"):
            item.setdefault(key, None)
        item = _normalize_task(item)
        signature = tuple(str(item.get(key) or "").strip().casefold() for key in ("title", "content", "deadline"))
        if signature not in seen:
            seen.add(signature) # pyright: ignore[reportArgumentType]
            result.append(item)
    if not result:
        raise ValueError("AI kh\u00f4ng nh\u1eadn di\u1ec7n \u0111\u01b0\u1ee3c task n\u00e0o trong c\u00e1c v\u00f9ng \u1ea3nh")
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

    return added


if __name__ == "__main__":
    image_path = Path(__file__).with_name("bt2.jpg")
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(exc)
