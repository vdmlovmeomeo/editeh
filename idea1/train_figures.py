"""
train_figures.py - "Dạy" model vẽ hình.

Nhập đề -> xem model trả JSON gì + hình vẽ ra -> đúng thì lưu, sai thì dán JSON đúng vào rồi lưu.
Ví dụ lưu trong examples.jsonl (cùng thư mục với main.py). Lần sau hỏi đề giống vậy,
chương trình sẽ đưa ví dụ đó cho model, nên model làm đúng dần.

Chạy: python train_figures.py

Lệnh:  :l  liệt kê ví dụ     :x  xoá ví dụ cuối     :seed  nạp các ca trong train_cases.txt + ví dụ mẫu mới vào examples.jsonl     :q  thoát
Hoặc mở examples.jsonl, mỗi dòng 1 ví dụ: {"problem": "...", "output": {...}}  -> sửa tay cũng được.
"""
import json

import main as d

HELP = """=== Dạy model vẽ hình ===
Gõ 1 đề để thử. Sau khi xem hình:
  y      = đúng, lưu thành ví dụ
  e      = sai, dán JSON đúng vào (1 dòng) rồi lưu
  Enter  = bỏ qua
Lệnh: :l liệt kê ví dụ | :x xoá ví dụ cuối | :seed thêm ví dụ mẫu mới | :q thoát"""


def show(data, prob):
    print("  JSON:", json.dumps(data, ensure_ascii=False))
    try:
        d.draw(d.parse(data), prob)
        return True
    except Exception as e:
        print("  lỗi khi vẽ:", repr(e))
        return False


def ask_json():
    while True:
        s = input("  Dán JSON (1 dòng, Enter trống = huỷ): ").strip()
        if not s:
            return None
        try:
            return d.extract_json(s)
        except Exception as e:
            print("  JSON sai:", e)


def main():
    d.ensure_examples_file()
    print(HELP)
    print(f"({len(d.load_examples())} ví dụ trong {d.EXAMPLES_PATH.name})")
    while True:
        prob = input("\nĐề: ").strip()
        if not prob:
            continue
        if prob == ":q":
            break
        if prob == ":l":
            for i, (p, _) in enumerate(d.load_examples(), 1):
                print(f"  {i:3}. {p}")
            continue
        if prob == ":seed":
            exs = d.load_examples()
            have = {p for p, _ in exs}
            new = [(p, o) for p, o in d.SEED_EXAMPLES if p not in have]
            d.save_examples(exs + new)
            print(f"  đã thêm {len(new)} ví dụ mẫu mới (ví dụ của bạn giữ nguyên)")
            continue
        if prob == ":x":
            exs = d.load_examples()
            if exs:
                d.save_examples(exs[:-1])
                print("  đã xoá:", exs[-1][0])
            continue

        data, ok = None, False
        try:
            data = d.extract_json(d.ask_model(prob, 0))
            ok = show(data, prob)
        except Exception as e:
            print("  model lỗi:", repr(e))

        while True:
            c = input("  [y] đúng, lưu | [e] sửa JSON | [Enter] bỏ qua: ").strip().lower()
            if c == "y":
                if data and ok:
                    d.add_example(prob, data)
                    print("  đã lưu.")
                    break
                print("  chưa có JSON vẽ được, dùng e để dán JSON đúng.")
            elif c == "e":
                nd = ask_json()
                if nd:
                    data = nd
                    ok = show(data, prob)
            else:
                break


if __name__ == "__main__":
    main()