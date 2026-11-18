from pathlib import Path
from ai_input import extract_tasks

# class Input():
#     ...

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    image_path = Path(__file__).with_name("testimg.png")
    try:
        print(extract_tasks(str(image_path)))
    except Exception as exc:
        print(f"Lỗi: {exc}")
