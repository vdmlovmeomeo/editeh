"""
draw_ai_v4.py - Vẽ hình từ đề bài: model (Ollama, local) chỉ TRÍCH XUẤT dữ kiện,
code tự tính tọa độ rồi vẽ bằng matplotlib.

Cài:   pip install requests matplotlib
Chạy:  python draw_ai_v4.py
       python draw_ai_v4.py "Tam giác ABC vuông tại A, AB=3, AC=4"
"""
import json
import math
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.patches import Circle, Polygon
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "falcon3:3b"
MAX_RETRY = 3

SYSTEM_PROMPT = """You convert a geometry problem (Vietnamese or English) into JSON.
DO NOT compute coordinates. Only extract the figure type, point names and given lengths. A program does the geometry.

Output ONLY one JSON object. Fields:
- "figure": triangle | right_triangle | isosceles_triangle | equilateral_triangle | rectangle | square | right_trapezoid | circle
- "names": vertex names in order, e.g. "ABC" or "ABCD"
- given lengths as two-letter keys, e.g. "AB": 3, "AC": 4
- "right_at": vertex with the right angle (right_triangle)
- "apex" and "height" (or the leg length) for isosceles_triangle
- "side": side length (equilateral_triangle, square)
- right_trapezoid: bases are AB and CD, AD is the height (right angles at A and D)
- circle: "center" and "r". If the radius is a letter (R, a...), put that letter as a string, e.g. "r":"R". NEVER invent a number.
- "extras": extra points, each one of
  {"name":"M","type":"midpoint","of":"BC"}
  {"name":"H","type":"foot","from":"A","to":"BC"}     (foot of the perpendicular from A to line BC)
  {"name":"G","type":"centroid","of":"ABC"}
  {"name":"O","type":"circumcircle","of":"ABC"}       (circumcenter + circumscribed circle)
  {"name":"N","type":"on_segment","of":"BC"}         (any point lying on segment BC)
- "on_circle": names of points lying on the circle, e.g. ["A","B"] (circle only). "(O;R)" means the circle with center O and radius R.
- "draw": extra segments to draw ONLY if the problem asks (nối, dây, bán kính...), e.g. ["AM"]

Problem: Hình chữ nhật MNPQ có MN=7, MQ=2
{"figure":"rectangle","names":"MNPQ","MN":7,"MQ":2}

Problem: Tam giác DEF vuông tại D, DE=6, DF=8, K là chân đường cao từ D xuống EF
{"figure":"right_triangle","names":"DEF","right_at":"D","DE":6,"DF":8,"extras":[{"name":"K","type":"foot","from":"D","to":"EF"}]}

Problem: Hình thang vuông ABCD vuông tại A và D, AB=5, CD=9, AD=4
{"figure":"right_trapezoid","names":"ABCD","AB":5,"CD":9,"AD":4}

Problem: Tam giác XYZ có XY=4, YZ=5, ZX=6, I là trung điểm YZ, nối X với I
{"figure":"triangle","names":"XYZ","XY":4,"YZ":5,"ZX":6,"extras":[{"name":"I","type":"midpoint","of":"YZ"}],"draw":["XI"]}

Problem: Tam giác PQR cân tại P, QR=8, đường cao PH=3
{"figure":"isosceles_triangle","names":"PQR","apex":"P","QR":8,"height":3,"extras":[{"name":"H","type":"foot","from":"P","to":"QR"}]}

Problem: Đường tròn tâm I bán kính 5
{"figure":"circle","center":"I","r":5}

Problem: Đường tròn tâm T bán kính a
{"figure":"circle","center":"T","r":"a"}

Problem: Cho đường tròn (S;b), M và N thuộc (S;b), vẽ dây MN
{"figure":"circle","center":"S","r":"b","on_circle":["M","N"],"draw":["MN"]}

Problem: Tam giác GHK vuông tại G, GH=6, GK=8, N thuộc HK
{"figure":"right_triangle","names":"GHK","right_at":"G","GH":6,"GK":8,"extras":[{"name":"N","type":"on_segment","of":"HK"}]}

Problem: Tam giác ABC vuông tại A, AB=5, AC=12, vẽ đường tròn ngoại tiếp
{"figure":"right_triangle","names":"ABC","right_at":"A","AB":5,"AC":12,"extras":[{"name":"O","type":"circumcircle","of":"ABC"}]}"""


# ---------------------------------------------------------------- model

def ask_model(prompt: str, temp: float) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": "Problem: " + prompt,
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": temp, "num_ctx": 4096, "num_predict": 1000},
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


# ---------------------------------------------------------------- hình học

def mid(p, q):
    return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)


def foot(p, a, b):
    """Hình chiếu vuông góc của p lên đường thẳng ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)
    return (a[0] + t * dx, a[1] + t * dy)


def circumcenter(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-9:
        raise ValueError("3 điểm thẳng hàng")
    sa, sb, sc = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    return (ux, uy)


def need(v, what):
    if v is None:
        raise ValueError("Thiếu " + what)
    return v


def root(x, what="số liệu"):
    if x <= 1e-9:
        raise ValueError("Không hợp lệ: " + what)
    return math.sqrt(x)


def get_sides(data: dict) -> dict:
    src = {**data, **(data.get("sides") or {})}
    out = {}
    for k, v in src.items():
        if isinstance(k, str) and re.fullmatch(r"[A-Z]{2}", k):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


# Mỗi builder nhận (names, S, data), trả về dict {tên điểm: (x, y)}
# S(p, q) = độ dài cạnh pq nếu đề cho, không thì None.

def b_triangle(n, S, data):
    A, B, C = n[:3]
    c, b, a = need(S(A, B), "AB"), need(S(A, C), "AC"), need(S(B, C), "BC")
    x = (b * b + c * c - a * a) / (2 * c)
    y = root(b * b - x * x, "3 cạnh không tạo thành tam giác")
    return {A: (0.0, 0.0), B: (c, 0.0), C: (x, y)}


def b_right_triangle(n, S, data):
    A, B, C = n[:3]
    r = str(data.get("right_at") or A)
    if r not in (A, B, C):
        raise ValueError("right_at không nằm trong names")
    p, q = [x for x in (A, B, C) if x != r]
    l1, l2, h = S(r, p), S(r, q), S(p, q)
    if l1 is None and l2 and h:
        l1 = root(h * h - l2 * l2)
    elif l2 is None and l1 and h:
        l2 = root(h * h - l1 * l1)
    if not (l1 and l2):
        raise ValueError("Thiếu độ dài 2 cạnh góc vuông")
    P = {r: (0.0, 0.0), p: (l1, 0.0), q: (0.0, l2)}
    return {x: P[x] for x in (A, B, C)}


def b_isosceles(n, S, data):
    A, B, C = n[:3]
    a = str(data.get("apex") or A)
    if a not in (A, B, C):
        raise ValueError("apex không nằm trong names")
    p, q = [x for x in (A, B, C) if x != a]
    base = need(S(p, q), "cạnh đáy")
    h = data.get("height")
    if h is None:
        leg = need(S(a, p) or S(a, q), "cạnh bên hoặc đường cao")
        h = root(leg * leg - base * base / 4, "cạnh bên quá ngắn")
    P = {p: (-base / 2, 0.0), q: (base / 2, 0.0), a: (0.0, float(h))}
    return {x: P[x] for x in (A, B, C)}


def b_equilateral(n, S, data):
    A, B, C = n[:3]
    s = float(need(data.get("side") or S(A, B) or S(B, C) or S(A, C), "cạnh"))
    return {A: (0.0, 0.0), B: (s, 0.0), C: (s / 2, s * math.sqrt(3) / 2)}


def b_rectangle(n, S, data):
    A, B, C, D = n[:4]
    w = need(S(A, B) or S(C, D), "chiều dài")
    h = need(S(A, D) or S(B, C), "chiều rộng")
    return {A: (0.0, 0.0), B: (w, 0.0), C: (w, h), D: (0.0, h)}


def b_square(n, S, data):
    A, B, C, D = n[:4]
    s = float(need(data.get("side") or S(A, B) or S(B, C) or S(C, D) or S(A, D), "cạnh"))
    return {A: (0.0, 0.0), B: (s, 0.0), C: (s, s), D: (0.0, s)}


def b_right_trapezoid(n, S, data):
    # vuông tại A và D: đáy AB, CD; đường cao AD
    A, B, C, D = n[:4]
    ab, cd, ad = S(A, B), S(C, D), S(A, D)
    if ad is None and ab and cd and S(B, C):
        ad = root(S(B, C) ** 2 - (cd - ab) ** 2, "BC quá ngắn")
    ab, cd, ad = need(ab, "AB"), need(cd, "CD"), need(ad, "AD")
    return {A: (0.0, 0.0), B: (ab, 0.0), C: (cd, ad), D: (0.0, ad)}


BUILDERS = {
    "triangle": b_triangle,
    "right_triangle": b_right_triangle,
    "isosceles_triangle": b_isosceles,
    "equilateral_triangle": b_equilateral,
    "rectangle": b_rectangle,
    "square": b_square,
    "right_trapezoid": b_right_trapezoid,
}


def build_figure(data: dict) -> dict:
    fig = data.get("figure")
    if fig != "circle" and fig not in BUILDERS:
        raise ValueError("Loại hình chưa hỗ trợ: " + str(fig))

    names = [str(c) for c in (data.get("names") or "") if str(c).strip()]
    sides = get_sides(data)
    S = lambda p, q: sides.get(p + q) or sides.get(q + p)
    shapes, segs, texts = [], [], []

    try:
        if fig == "circle":
            o = str(data.get("center") or "O")
            raw_r = data.get("r") or data.get("radius")
            if raw_r is None:
                raise ValueError("Thiếu bán kính")
            try:
                r = float(raw_r)
                label = f"{r:g}"
            except (TypeError, ValueError):  # bán kính là chữ (R, a...) -> vẽ cỡ mặc định
                r, label = 3.0, str(raw_r)
            pts = {o: (0.0, 0.0)}
            shapes.append({"type": "circle", "cx": 0.0, "cy": 0.0, "r": r,
                           "color": "tab:blue", "fill": False})
            # điểm thuộc đường tròn: đặt ở các góc cố định, tránh trùng nhau
            on = [str(x) for x in list(data.get("on_circle") or [])]
            k = len(on)
            angs = {1: [90], 2: [150, 30], 3: [150, 30, 270]}.get(k) or \
                   [150 - i * 360 / k for i in range(k)]
            for nm, a in zip(on, angs):
                pts[nm] = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
            # bán kính vẽ tới điểm đầu tiên thuộc đường tròn (không có thì vẽ nằm ngang)
            tx, ty = pts[on[0]] if on else (r, 0.0)
            segs.append([(0.0, 0.0), (tx, ty)])
            texts.append({"x": tx / 2 - ty / r * 0.07 * r, "y": ty / 2 + tx / r * 0.07 * r,
                          "text": label})
        else:
            pts = BUILDERS[fig](names, S, data)
            segs.append(list(pts.values()) + [next(iter(pts.values()))])

            # ghi số đo các cạnh mà đề cho, đặt lệch ra phía ngoài hình
            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            size = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            for k, v in sides.items():
                p, q = k[0], k[1]
                if p in pts and q in pts and p != q:
                    mx, my = mid(pts[p], pts[q])
                    dx, dy = mx - cx, my - cy
                    d = math.hypot(dx, dy) or 1.0
                    texts.append({"x": mx + dx / d * 0.07 * size,
                                  "y": my + dy / d * 0.07 * size, "text": f"{v:g}"})

        for e in data.get("extras") or []:
            t, name = e.get("type"), str(e.get("name") or "")
            of = list(e.get("of") or "")
            if t == "midpoint":
                pts[need(name or None, "tên điểm")] = mid(pts[of[0]], pts[of[1]])
            elif t == "foot":
                frm, to = str(e["from"]), list(e["to"])
                pts[need(name or None, "tên điểm")] = foot(pts[frm], pts[to[0]], pts[to[1]])
                segs.append([pts[frm], pts[name]])
            elif t == "on_segment":
                p, q = pts[of[0]], pts[of[1]]
                k = float(e.get("ratio") or 0.4)
                pts[need(name or None, "tên điểm")] = (p[0] + k * (q[0] - p[0]),
                                                       p[1] + k * (q[1] - p[1]))
            elif t == "centroid":
                P = [pts[x] for x in of[:3]]
                pts[need(name or None, "tên điểm")] = (
                    sum(p[0] for p in P) / 3, sum(p[1] for p in P) / 3)
            elif t == "circumcircle":
                name = name or "O"
                P = [pts[x] for x in of[:3]]
                o = circumcenter(*P)
                pts[name] = o
                shapes.append({"type": "circle", "cx": o[0], "cy": o[1],
                               "r": math.dist(o, P[0]), "color": "tab:green", "fill": False})
            else:
                raise ValueError("extras chưa hỗ trợ: " + str(t))

        for s in data.get("draw") or []:
            chain = [pts[x] for x in list(s) if x in pts]
            if len(chain) >= 2:
                segs.append(chain)
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("Model trả thiếu/sai dữ liệu: " + repr(e))

    return {"points": pts, "segments": segs, "texts": texts, "shapes": shapes}


# Dự phòng: nếu model trả thẳng tọa độ (points/segments/shapes) thì vẽ luôn
def build_raw(data: dict) -> dict:
    pts = {str(k): (float(v[0]), float(v[1])) for k, v in (data.get("points") or {}).items()}
    segs = []
    for chain in data.get("segments") or []:
        coords = [pts[n] for n in chain if n in pts]
        if len(coords) >= 2:
            segs.append(coords)
    texts = [{"x": float(t["x"]), "y": float(t["y"]), "text": str(t["text"])}
             for t in (data.get("texts") or [])]
    shapes = []
    for s in data.get("shapes") or []:
        color = s.get("color", "tab:blue")
        color = color if is_color_like(color) else "tab:blue"
        if s.get("type") == "circle":
            shapes.append({"type": "circle", "cx": float(s["cx"]), "cy": float(s["cy"]),
                           "r": float(s["r"]), "color": color, "fill": bool(s.get("fill"))})
    if not (pts or segs or shapes):
        raise ValueError("Không có gì để vẽ")
    return {"points": pts, "segments": segs, "texts": texts, "shapes": shapes}


def parse(data: dict) -> dict:
    return build_figure(data) if data.get("figure") else build_raw(data)


# ---------------------------------------------------------------- vẽ

def draw(d: dict, title: str):
    fig, ax = plt.subplots(figsize=(7, 7))

    for s in d["shapes"]:
        ax.add_patch(Circle((s["cx"], s["cy"]), s["r"], fill=s["fill"],
                            facecolor=s["color"] if s["fill"] else "none",
                            edgecolor=s["color"], lw=2))

    for chain in d["segments"]:
        xs, ys = zip(*chain)
        ax.plot(xs, ys, color="black", lw=2)

    if d["points"]:
        cx = sum(p[0] for p in d["points"].values()) / len(d["points"])
        cy = sum(p[1] for p in d["points"].values()) / len(d["points"])
        for name, (x, y) in d["points"].items():
            ax.plot(x, y, "o", color="black", ms=5)
            dx, dy = x - cx, y - cy
            n = math.hypot(dx, dy)
            if n < 1e-9:
                dx, dy, n = -1.0, -1.0, math.sqrt(2)
            ax.annotate(name, (x, y), xytext=(dx / n * 14, dy / n * 14),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=14, fontweight="bold")

    for t in d["texts"]:
        ax.plot(t["x"], t["y"], alpha=0)  # để autoscale tính cả vị trí text
        ax.text(t["x"], t["y"], t["text"], ha="center", va="center",
                fontsize=12, color="tab:red")

    ax.autoscale_view()
    ax.margins(0.2)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=10)
    plt.show()


def run(prompt: str):
    last = None
    for i in range(1, MAX_RETRY + 1):
        temp = 0 if i == 1 else 0.5  # temp=0 thì retry ra y hệt, nên các lần sau tăng nhẹ
        print(f"[{i}/{MAX_RETRY}] đang hỏi {MODEL}...")
        try:
            raw = ask_model(prompt, temp)
            print("  model trả:", raw.strip()[:500])
            draw(parse(extract_json(raw)), prompt)
            return
        except Exception as e:
            last = e
            print("  lỗi:", repr(e))
    print("Thất bại:", last)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        while True:
            p = input("Đề: ").strip()
            if p.lower() == 'exit':
                break
            run(p)