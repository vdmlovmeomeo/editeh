import json
import re
from datetime import date
import ollama # pyright: ignore[reportMissingImports]
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

PROMPT = """
Đọc các dòng bài tập viết tay trong ảnh. Chỉ trả về JSON array, không giải thích.
Mỗi dòng bài tập là một task riêng, không được bỏ dòng nào. Bỏ qua phần ghi chú đầu trang như Date, Sĩ số, Vắng, Muộn và tiêu đề Tuần.
Schema: {"title": "môn học hoặc tên việc", "content": "việc cần làm", "deadline": "YYYY-MM-DD" hoặc null}.
QUY TẮC TÁCH TRƯỜNG:
- Nếu dòng có dạng "Môn học: việc cần làm", title chỉ là môn học; content chỉ là việc cần làm. Ví dụ "Văn: Hoàn thành HDOT Ngữ văn (làm ra giấy)" => title="Văn", content="Hoàn thành HDOT Ngữ văn (làm ra giấy)".
- Nếu dòng chỉ ghi tên một việc/sự kiện, không có môn học hay phần việc riêng, đặt toàn bộ tên đó vào title và content là chuỗi rỗng. Ví dụ "MẶC LỄ PHỤC" => title="Mặc lễ phục", content="".
- Nếu dòng có môn học rồi đến tên bài/việc, tách môn học làm title và tên bài/việc làm content. Ví dụ "Toán HDOT" => title="Toán", content="HDOT"; "Phiếu mục tiêu GHKI" => title="Phiếu mục tiêu", content="GHKI".
- Đọc toàn bộ chiều ngang của từng dòng, đặc biệt cột bên phải. Chữ viết ở cột bên phải thuộc deadline hoặc ghi chú của task cùng hàng, không nối vào title/content. T2..T7, CN là thứ trong tuần, tuyệt đối không phải title. Hãy chuyển T2..T7/CN thành ngày cụ thể gần nhất trong năm hiện tại; nếu có "(tuần sau)" thì cộng thêm 7 ngày. Không trả deadline=null khi ảnh có ghi thứ; chỉ trả null khi dòng đó thật sự không có ngày/thứ.
- Trước khi trả JSON, tự kiểm tra từng dòng: nếu nhìn thấy T2, T3, T4, T5, T6, T7 hoặc CN ở bên phải thì bắt buộc điền deadline cho task tương ứng.
Ngày D/M hoặc DD/MM là ngày/tháng. Dùng năm hiện tại nếu ngày đó chưa qua; nếu đã qua thì dùng năm kế tiếp. Ngày ghi rõ năm thì giữ nguyên năm.
Không tự bịa chữ khó đọc; giữ nguyên phần đọc được. Chỉ dùng đúng các key title, content, deadline."""
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
            ''' A weekday plus day/month identifies the intended school year. Pick
                the matching weekday closest to today, allowing a date just passed.'''
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
    '''Weekday notes are often returned in deadline, or appended to the right
        of the task text. Resolve them before validating explicit date strings.'''
    for key in ("deadline", "title", "content"):
        value = str(item.get(key) or "")
        weekday = _weekday_deadline(value)
        if weekday and not item.get("deadline"):
            item["deadline"] = weekday.isoformat()
        if key != "deadline" and weekday:
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
            ''' Old years are usually vision hallucinations (the board shows
                the current school year). Do not expose a stale deadline.'''
            if parsed_deadline.year < date.today().year:
                item["deadline"] = None
            else:
                item["deadline"] = parsed_deadline.isoformat()
        except ValueError:
            pass
    return item


def _image_strips(image: Path) -> list[Path]:
    """Drop the page heading and split the task area into two overlapping crops."""
    paths: list[Path] = []
    with Image.open(image) as original:
        source = original.convert("RGB")
        width, height = source.size
        top = int(height * 0.13) if height > width * 0.55 else 0
        usable = height - top
        strip_height = max(usable // 2, 1)
        starts = [top, top + max(strip_height - STRIP_OVERLAP, 1)]
        for start in starts:
            end = min(start + strip_height, height)
            crop = source.crop((0, start, width, end))
            crop.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            temp = NamedTemporaryFile(suffix=".jpg", delete=False)
            path = Path(temp.name)
            temp.close()
            crop.save(path, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
            paths.append(path)
            if end == height:
                break
    return paths


def extract_tasks(image_path: str) -> list[dict]:
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Kh\u00f4ng t\u00ecm th\u1ea5y \u1ea3nh: {image}")

    # Send the original board as one image; do not split it into strips.
    prepared_images = [_prepare_image(image)]
    today = date.today()
    weekday_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    example_output_JSON = """[
  {"title": "ESL", "content": "Language Companion", "deadline": "06/07/2026"},
  {"title": "ESL", "content": "Flyer", "deadline": "06/07/2026"},
  {"title": "ESL", "content": "Workbook p42", "deadline": "06/07/2026"}
]
"""
    date_context = (
        f"\nTHÔNG TIN THỜI GIAN THAM CHIẾU: Hôm nay là {weekday_names[today.weekday()]} "
        f"ngày {today.day:02d}/{today.month:02d}/{today.year}. "
        "Dùng mốc này để đổi T2..T7/CN thành ngày ISO YYYY-MM-DD. "
        "QUAN TRỌNG: phải quét từ trên xuống dưới toàn bộ ảnh và trả về một phần tử cho MỖI dòng homework; "
        "không dừng sau dòng đầu tiên, kể cả khi dòng đó có content rỗng."
    )
    client = ollama.Client(host="http://localhost:11434", timeout=300)
    tasks: list[dict] = []
    try:
        for prepared_image in prepared_images:
            response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "Read every visible homework in the image from top to bottom. Each line is a separate task; never merge adjacent lines. The title is only subject like (e.g. toán, văn, tiếng Anh, lý, hóa, sinh, sử, địa, tin,..., esl, math, literature...). Content is the work text (workbook, language companion, HDOT, SBT,...). If the subject is repeated for subsequent lines, use the repeated subject. Return ONLY a JSON array, one object of the JSON array per row of infomation in the image, using exactly 'title', 'content', 'deadline'.\n" + date_context + f"Your output must look like this: {example_output_JSON}", "images": [str(prepared_image.resolve())]}],
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
                # Some vision models wrap the array in a named property.
                wrapped = next(
                    (parsed[key] for key in ("tasks", "items", "data", "results")
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

    manager.save_to_csv()
    return added

if __name__ == "__main__":
    image_path = Path(__file__).with_name("bt.jpg")#ctx:4282tk
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(exc)
