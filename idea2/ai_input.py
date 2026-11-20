import json
import re
from datetime import date
import ollama # pyright: ignore[reportMissingImports]
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

PROMPT = """
\u0110\u1ecd c\u00e1c d\u00f2ng b\u00e0i t\u1eadp vi\u1ebft tay trong \u1ea3nh. Ch\u1ec9 tr\u1ea3 v\u1ec1 JSON array, kh\u00f4ng gi\u1ea3i th\u00edch.
M\u1ed7i d\u00f2ng m\u00f4n h\u1ecdc l\u00e0 m\u1ed9t task ri\u00eang, kh\u00f4ng \u0111\u01b0\u1ee3c b\u1ecf c\u00e1c d\u00f2ng \u1edf d\u01b0\u1edbi. B\u1ecf qua ti\u00eau \u0111\u1ec1 chung \u0111\u1ea7u trang.
Schema: {"title": "m\u00f4n h\u1ecdc", "content": "vi\u1ec7c c\u1ea7n l\u00e0m", "deadline": "YYYY-MM-DD" ho\u1eb7c null}.
T\u00e1ch d\u1ea1ng "M\u00f4n: vi\u1ec7c" th\u00e0nh title v\u00e0 content. Ghi ch\u00fa ng\u00e0y/th\u1ee9 \u1edf c\u1ed9t b\u00ean ph\u1ea3i thu\u1ed9c v\u1ec1 task c\u00f9ng h\u00e0ng.
T2..T7, CN l\u00e0 th\u1ee9 trong tu\u1ea7n, kh\u00f4ng ph\u1ea3i title. "tu\u1ea7n sau" kh\u00f4ng ph\u1ea3i ng\u00e0y c\u1ee5 th\u1ec3 n\u00ean deadline null.
Ng\u00e0y D/M ho\u1eb7c DD/MM l\u00e0 ng\u00e0y/th\u00e1ng. D\u00f9ng n\u0103m hi\u1ec7n t\u1ea1i n\u1ebfu ng\u00e0y \u0111\u00f3 c\u00f2n \u1edf ph\u00eda tr\u01b0\u1edbc, n\u1ebfu \u0111\u00e3 qua th\u00ec d\u00f9ng n\u0103m k\u1ebf ti\u1ebfp.
Kh\u00f4ng t\u1ef1 b\u1ecba ch\u1eef kh\u00f3 \u0111\u1ecdc; gi\u1eef nguy\u00ean ph\u1ea7n \u0111\u1ecdc \u0111\u01b0\u1ee3c. Ch\u1ec9 d\u00f9ng \u0111\u00fang c\u00e1c key title, content, deadline."""

MODEL = "gemma3n:e2b"
MAX_IMAGE_SIDE = 1800
IMAGE_QUALITY = 85
STRIP_OVERLAP = 100


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
    if item.get("deadline"):
        try:
            item["deadline"] = parse_deadline(str(item["deadline"])).isoformat()
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

    prepared_images = _image_strips(image)
    client = ollama.Client(host="http://localhost:11434", timeout=300)
    tasks: list[dict] = []
    try:
        for prepared_image in prepared_images:
            response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT, "images": [str(prepared_image.resolve())]}],
                format="json",
                options={"temperature": 0, "num_predict": 768, "num_ctx": 8192},
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
                parsed = [parsed]
            if isinstance(parsed, list):
                tasks.extend(item for item in parsed if isinstance(item, dict))
    except Exception as exc:
        raise RuntimeError(f"Kh\u00f4ng g\u1ecdi \u0111\u01b0\u1ee3c Ollama: {type(exc).__name__}: {exc}") from exc
    finally:
        for prepared_image in prepared_images:
            prepared_image.unlink(missing_ok=True)

    # Dedupe items repeated in the overlap between crops.
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
    image_path = Path(__file__).with_name("realsamplecroped.jpg")#ctx:4282tk
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(f"Lỗi: {exc}")
