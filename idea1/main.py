"""
draw_ai_v2.py - Vẽ hình từ prompt bằng Ollama (local) + matplotlib

Cài:   pip install requests matplotlib
Chạy:  python draw_ai_v2.py                  (nhập prompt, 'q' để thoát)
       python draw_ai_v2.py "hình chữ nhật 3x2"
"""
import json
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "falcon3:3b"
MAX_RETRY = 3

SYSTEM_PROMPT = """You convert a drawing request into JSON shapes.
Use the REAL numbers from the request as sizes (3x2 means width 3, height 2). Do NOT rescale.
If no size is given, use small simple numbers (like 1 to 10). Origin (0,0) is bottom-left, y goes up.

Output ONLY one JSON object, no explanation. Shape types:
- {"type":"rect","x":0,"y":0,"w":W,"h":H}                      (x,y = bottom-left corner)
- {"type":"circle","cx":0,"cy":0,"r":R}
- {"type":"poly","points":[[x,y],...],"closed":true}          (closed=false for a line)
Every shape can also have "color" and "fill" (true/false).

Request: rectangle 5x3
{"shapes":[{"type":"rect","x":0,"y":0,"w":5,"h":3}]}

Request: right triangle with legs 4 and 3, and a circle of radius 1 next to it
{"shapes":[{"type":"poly","points":[[0,0],[4,0],[0,3]],"closed":true,"color":"blue"},{"type":"circle","cx":6,"cy":1,"r":1,"color":"red"}]}"""


def ask_model(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": "Request: " + prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1500},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("Không tìm thấy JSON")
    return json.loads(text[a : b + 1])


def validate(data: dict) -> list:
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("Thiếu 'shapes'")
    out = []
    for s in shapes:
        t = s.get("type")
        if t == "rect":
            x, y = float(s.get("x", 0)), float(s.get("y", 0))
            w, h = float(s["w"]), float(s["h"])
            out.append({**s, "type": "poly", "closed": True,
                        "points": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]})
        elif t == "circle":
            out.append({**s, "cx": float(s["cx"]), "cy": float(s["cy"]), "r": float(s["r"])})
        elif t == "poly":
            pts = [(float(p[0]), float(p[1])) for p in s["points"]]
            if len(pts) >= 2:
                out.append({**s, "points": pts})
    if not out:
        raise ValueError("Không có shape hợp lệ")
    return out


def draw(shapes: list, title: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    for s in shapes:
        color = s.get("color", "tab:blue")
        fill = bool(s.get("fill", False))
        kw = dict(fill=fill, facecolor=color if fill else "none",
                  edgecolor="black" if fill else color, lw=2)
        try:
            if s["type"] == "circle":
                ax.add_patch(Circle((s["cx"], s["cy"]), s["r"], **kw))
            elif s.get("closed", False):
                ax.add_patch(Polygon(s["points"], closed=True, **kw))
            else:
                xs, ys = zip(*s["points"])
                ax.plot(xs, ys, color=color, lw=2)
        except ValueError:  # tên màu sai -> đổi sang màu mặc định
            s["color"] = "tab:blue"
            kw.update(facecolor="none", edgecolor="tab:blue")
            if s["type"] == "circle":
                ax.add_patch(Circle((s["cx"], s["cy"]), s["r"], **kw))
            else:
                ax.add_patch(Polygon(s["points"], closed=s.get("closed", False), **kw))

    ax.autoscale_view()
    ax.margins(0.15)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    plt.show()


def run(prompt: str):
    last = None
    for i in range(1, MAX_RETRY + 1):
        print(f"[{i}/{MAX_RETRY}] đang hỏi {MODEL}...")
        try:
            raw = ask_model(prompt)
            print("  model trả:", raw.strip()[:300])
            draw(validate(extract_json(raw)), prompt)
            return
        except Exception as e:
            last = e
            print("  lỗi:", e)
    print("Thất bại:", last)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        while True:
            p = input("Prompt (q để thoát): ").strip()
            if p.lower() in ("q", "quit", ""):
                break
            run(p)