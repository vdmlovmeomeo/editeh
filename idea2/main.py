from pathlib import Path
from ai_input import extract_tasks
# class Input():
#     ...

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8") # type: ignore
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("realsamplecroped.jpg")
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(exc)
