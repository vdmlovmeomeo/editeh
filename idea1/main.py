import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.patches import Circle
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "falcon3:7b"
MAX_RETRY = 3
N_EXAMPLES = 6  # số ví dụ giống nhất đưa vào prompt
EXAMPLES_PATH = Path(__file__).with_name("examples.jsonl")

PROMPT_HEAD = """You convert a geometry problem (Vietnamese or English) into JSON.
DO NOT compute coordinates and NEVER invent numbers. Only extract the figure type, the point names and the lengths the problem actually gives. A program does the geometry and draws a default shape when no length is given.

Output ONLY one JSON object. Fields:
- "figure": triangle | right_triangle | right_isosceles_triangle | isosceles_triangle | equilateral_triangle | rectangle | square | parallelogram | rhombus | trapezoid | isosceles_trapezoid | right_trapezoid | quadrilateral | regular_polygon | circle | none (only lines and points, no polygon or circle)
- "names": vertex names in order, e.g. "ABC", "ABCD", "ABCDEF"
- given lengths as two-letter keys, ONLY if the problem gives numbers, e.g. "AB": 3, "AC": 4
- "right_at": vertex with the right angle (right_triangle, right_isosceles_triangle)
- "apex" (and "height" if given) for isosceles_triangle
- "side": side length (equilateral_triangle, square, rhombus, regular_polygon), only if given
- "angle": angle at the first vertex in degrees (parallelogram, rhombus), only if given
- trapezoid / isosceles_trapezoid / right_trapezoid: the bases are AB and CD (AB // CD); right_trapezoid has right angles at A and D and AD is the height; "height" only if given
- regular_polygon: "names" has one letter per vertex
- circle: "center"; "r" ONLY if a NUMBER is given (never put a letter in r). "(O;R)" means center O and radius R.
- "on_circle": names of the points lying on the circle, e.g. ["A","B"]
- "extras": extra points, each one of
  {"name":"M","type":"midpoint","of":"BC"}
  {"name":"H","type":"foot","from":"A","to":"BC"}     (foot of the perpendicular from A to line BC, or to a named line d)
  {"name":"N","type":"on_segment","of":"BC"}         (any point lying on segment BC)
  {"name":"G","type":"centroid","of":"ABC"}
  {"name":"O","type":"circumcircle","of":"ABC"}       (circumcenter + circumscribed circle)
  {"type":"line","name":"d","through":["A","B"]}       (line through two points)
  {"type":"line","name":"d","through":"M","parallel_to":"BC"}       (parallel_to / perpendicular_to take a line name or two point names)
  {"type":"line","name":"d","through":"M","perpendicular_to":"BC"}
  {"type":"line","name":"t","tangent_at":"A","of":"O"}    (tangent at A to the circle with center O)
  {"type":"line","name":"d"}                               (a free line, no "through")
  {"name":"A","type":"on_line","of":"d"}                   (point lying on line d)
  {"name":"M","type":"off_line","of":"d"}                  (point NOT lying on line d)
  {"name":"M","type":"intersection","of":["d","BC"]}       (intersection of two lines)
  extras are built in order: list a point before any line that uses it.
- "draw": segments to draw ONLY if the problem asks (nối, dây, bán kính, đường kính...), e.g. ["AM"], ["OA"]"""

# Ví dụ mẫu ban đầu. Lần chạy đầu tiên sẽ được ghi ra examples.jsonl, sau đó sửa file đó.
SEED_EXAMPLES = [
    ("Hình chữ nhật MNPQ có MN=7, MQ=2",
     {"figure": "rectangle", "names": "MNPQ", "MN": 7, "MQ": 2}),
    ("Hình chữ nhật EFGH", {"figure": "rectangle", "names": "EFGH"}),
    ("Hình vuông MNPQ cạnh 5", {"figure": "square", "names": "MNPQ", "side": 5}),
    ("Hình vuông EFGH", {"figure": "square", "names": "EFGH"}),
    ("Tam giác DEF vuông tại D, DE=6, DF=8, K là chân đường cao từ D xuống EF",
     {"figure": "right_triangle", "names": "DEF", "right_at": "D", "DE": 6, "DF": 8,
      "extras": [{"name": "K", "type": "foot", "from": "D", "to": "EF"}]}),
    ("Tam giác XYZ vuông tại X",
     {"figure": "right_triangle", "names": "XYZ", "right_at": "X"}),
    ("Tam giác MNP vuông cân tại M, MN=4",
     {"figure": "right_isosceles_triangle", "names": "MNP", "right_at": "M", "MN": 4}),
    ("Tam giác XYZ có XY=4, YZ=5, ZX=6, I là trung điểm YZ, nối X với I",
     {"figure": "triangle", "names": "XYZ", "XY": 4, "YZ": 5, "ZX": 6,
      "extras": [{"name": "I", "type": "midpoint", "of": "YZ"}], "draw": ["XI"]}),
    ("Tam giác MNP", {"figure": "triangle", "names": "MNP"}),
    ("Tam giác PQR cân tại P, QR=8, đường cao PH=3",
     {"figure": "isosceles_triangle", "names": "PQR", "apex": "P", "QR": 8, "height": 3,
      "extras": [{"name": "H", "type": "foot", "from": "P", "to": "QR"}]}),
    ("Tam giác UVW cân tại U",
     {"figure": "isosceles_triangle", "names": "UVW", "apex": "U"}),
    ("Tam giác đều DEF cạnh 6",
     {"figure": "equilateral_triangle", "names": "DEF", "side": 6}),
    ("Tam giác đều MNP", {"figure": "equilateral_triangle", "names": "MNP"}),
    ("Hình bình hành EFGH", {"figure": "parallelogram", "names": "EFGH"}),
    ("Hình bình hành MNPQ có MN=6, MQ=4, góc M = 70 độ",
     {"figure": "parallelogram", "names": "MNPQ", "MN": 6, "MQ": 4, "angle": 70}),
    ("Hình thoi EFGH", {"figure": "rhombus", "names": "EFGH"}),
    ("Hình thoi MNPQ cạnh 5, góc M = 60 độ",
     {"figure": "rhombus", "names": "MNPQ", "side": 5, "angle": 60}),
    ("Hình thang MNPQ (MN // PQ)", {"figure": "trapezoid", "names": "MNPQ"}),
    ("Hình thang cân MNPQ (MN // PQ), MN=8, PQ=4",
     {"figure": "isosceles_trapezoid", "names": "MNPQ", "MN": 8, "PQ": 4}),
    ("Hình thang vuông ABCD vuông tại A và D, AB=5, CD=9, AD=4",
     {"figure": "right_trapezoid", "names": "ABCD", "AB": 5, "CD": 9, "AD": 4}),
    ("Hình thang vuông EFGH vuông tại E và H",
     {"figure": "right_trapezoid", "names": "EFGH"}),
    ("Tứ giác EFGH", {"figure": "quadrilateral", "names": "EFGH"}),
    ("Lục giác đều ABCDEF", {"figure": "regular_polygon", "names": "ABCDEF"}),
    ("Ngũ giác đều MNPQR cạnh 4",
     {"figure": "regular_polygon", "names": "MNPQR", "side": 4}),
    ("Đường tròn (I;R)", {"figure": "circle", "center": "I"}),
    ("Đường tròn tâm K bán kính 4", {"figure": "circle", "center": "K", "r": 4}),
    ("Cho (S;b), M và N thuộc (S;b), vẽ dây MN",
     {"figure": "circle", "center": "S", "on_circle": ["M", "N"], "draw": ["MN"]}),
    ("Cho (T;r), ba điểm M, N, P thuộc (T;r)",
     {"figure": "circle", "center": "T", "on_circle": ["M", "N", "P"]}),
    ("Đường tròn (I;R), M thuộc (I;R), nối I với M",
     {"figure": "circle", "center": "I", "on_circle": ["M"], "draw": ["IM"]}),
    ("Tam giác GHK vuông tại G, GH=6, GK=8, N thuộc HK",
     {"figure": "right_triangle", "names": "GHK", "right_at": "G", "GH": 6, "GK": 8,
      "extras": [{"name": "N", "type": "on_segment", "of": "HK"}]}),
    ("Tam giác DEF vuông tại D, DE=5, DF=12, vẽ đường tròn ngoại tiếp",
     {"figure": "right_triangle", "names": "DEF", "right_at": "D", "DE": 5, "DF": 12,
      "extras": [{"name": "O", "type": "circumcircle", "of": "DEF"}]}),
    ("Cho đường thẳng d và điểm A không thuộc d",
     {"figure": "none", "extras": [{"type": "line", "name": "d"},
                                   {"name": "A", "type": "off_line", "of": "d"}]}),
    ("Cho đường thẳng a, các điểm M, N thuộc a",
     {"figure": "none", "extras": [{"type": "line", "name": "a"},
                                   {"name": "M", "type": "on_line", "of": "a"},
                                   {"name": "N", "type": "on_line", "of": "a"}]}),
    ("Cho đường thẳng d và điểm K không thuộc d, kẻ KH vuông góc với d tại H",
     {"figure": "none", "extras": [{"type": "line", "name": "d"},
                                   {"name": "K", "type": "off_line", "of": "d"},
                                   {"name": "H", "type": "foot", "from": "K", "to": "d"}]}),
    ("Hai đường thẳng a và b cắt nhau tại O",
     {"figure": "none", "extras": [{"type": "line", "name": "a"}, {"type": "line", "name": "b"},
                                   {"name": "O", "type": "intersection", "of": ["a", "b"]}]}),
    ("Hai đường thẳng a và b song song",
     {"figure": "none", "extras": [{"type": "line", "name": "a"},
                                   {"type": "line", "name": "b", "parallel_to": "a"}]}),
    ("Tam giác MNP, đường thẳng d đi qua M và song song với NP",
     {"figure": "triangle", "names": "MNP",
      "extras": [{"type": "line", "name": "d", "through": "M", "parallel_to": "NP"}]}),
    ("Tam giác DEF, đường thẳng a đi qua D và vuông góc với EF",
     {"figure": "triangle", "names": "DEF",
      "extras": [{"type": "line", "name": "a", "through": "D", "perpendicular_to": "EF"}]}),
    ("Tam giác PQR, I thuộc PQ, K thuộc PR, đường thẳng d đi qua I và K",
     {"figure": "triangle", "names": "PQR",
      "extras": [{"name": "I", "type": "on_segment", "of": "PQ"},
                 {"name": "K", "type": "on_segment", "of": "PR"},
                 {"type": "line", "name": "d", "through": ["I", "K"]}]}),
    ("Đường thẳng m cắt đường tròn (S;R) tại U và V",
     {"figure": "circle", "center": "S", "on_circle": ["U", "V"],
      "extras": [{"type": "line", "name": "m", "through": ["U", "V"]}]}),
    ("Tiếp tuyến tại M của đường tròn (I;R)",
     {"figure": "circle", "center": "I", "on_circle": ["M"],
      "extras": [{"type": "line", "name": "t", "tangent_at": "M", "of": "I"}]}),
]


# ---------------------------------------------------------------- ví dụ mẫu

def save_examples(exs):
    with EXAMPLES_PATH.open("w", encoding="utf-8") as f:
        for p, o in exs:
            f.write(json.dumps({"problem": p, "output": o}, ensure_ascii=False) + "\n")


def ensure_examples_file():
    if not EXAMPLES_PATH.exists():
        save_examples(SEED_EXAMPLES)


def load_examples():
    ensure_examples_file()
    out = []
    for line in EXAMPLES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            out.append((str(e["problem"]), dict(e["output"])))
        except Exception:  # dòng hỏng thì bỏ qua
            continue
    return out


def add_example(problem, output):
    exs = [e for e in load_examples() if e[0] != problem]
    exs.append((problem, output))
    save_examples(exs)


def keywords(text):
    """Các từ khoá thường (bỏ tên điểm viết hoa như ABC, O, R và số)."""
    return {w.lower() for w in re.findall(r"[^\W\d_]+", text) if not w.isupper()}


def pick_examples(prompt, exs, k=N_EXAMPLES):
    q = keywords(prompt)
    scored = []
    for i, (p, o) in enumerate(exs):
        e = keywords(p)
        union = q | e
        scored.append((len(q & e) / len(union) if union else 0.0, i, p, o))
    best = sorted(scored, reverse=True)[:k]  # điểm bằng nhau -> ưu tiên ví dụ mới thêm
    best.reverse()  # giống nhất để cuối cùng, sát với đề cần hỏi
    return [(p, o) for _, _, p, o in best]


def build_system(prompt):
    parts = [PROMPT_HEAD]
    for p, o in pick_examples(prompt, load_examples()):
        parts.append("Problem: " + p + "\n" + json.dumps(o, ensure_ascii=False, separators=(",", ":")))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- model

def ask_model(prompt: str, temp: float) -> str:
    payload = {
        "model": MODEL,
        "system": build_system(prompt),
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


def nrm(v):
    """Vector pháp tuyến (quay v 90 độ)."""
    return (-v[1], v[0])


# Vị trí các điểm đặt lên/ngoài một đường thẳng (nhân với "scale" của đường đó)
ON_THROUGH = [1.0, -1.0, 1.5, -1.5, 2.0, -2.0]   # đường qua 2 điểm đã có: đặt ra ngoài 2 điểm đó
ON_FREE = [-0.5, 0.5, 0.0, -1.0, 1.0]            # đường tự do
OFF_SEQ = [0.0, 0.7, -0.7, 1.4, -1.4]


def fit(P, S):
    """Co giãn hình mặc định P cho khớp với cạnh đầu tiên mà đề có cho số đo."""
    names = list(P)
    for i in range(len(names)):
        p, q = names[i], names[(i + 1) % len(names)]
        s = S(p, q)
        if s:
            k = s / math.dist(P[p], P[q])
            return {x: (v[0] * k, v[1] * k) for x, v in P.items()}
    return P


def split_right(n, data):
    A, B, C = n[:3]
    r = str(data.get("right_at") or A)
    if r not in (A, B, C):
        raise ValueError("right_at không nằm trong names")
    p, q = [x for x in (A, B, C) if x != r]
    return (A, B, C), r, p, q


# Mỗi builder nhận (names, S, data), trả về dict {tên điểm: (x, y)}.
# S(p, q) = độ dài cạnh pq nếu đề có cho, không thì None. Thiếu số đo -> dùng cỡ mặc định.

def b_triangle(n, S, data):
    A, B, C = n[:3]
    c, b, a = S(A, B), S(A, C), S(B, C)
    if a and b and c:
        x = (b * b + c * c - a * a) / (2 * c)
        y = root(b * b - x * x, "3 cạnh không tạo thành tam giác")
        return {A: (0.0, 0.0), B: (c, 0.0), C: (x, y)}
    return fit({A: (0.0, 0.0), B: (5.0, 0.0), C: (1.8, 3.6)}, S)


def b_right_triangle(n, S, data):
    (A, B, C), r, p, q = split_right(n, data)
    l1, l2, h = S(r, p), S(r, q), S(p, q)
    if l1 is None and l2 and h:
        l1 = root(h * h - l2 * l2)
    elif l2 is None and l1 and h:
        l2 = root(h * h - l1 * l1)
    elif not l1 and not l2 and h:
        l1, l2 = 0.6 * h, 0.8 * h
    if l1 and not l2:
        l2 = l1 * 4 / 3
    elif l2 and not l1:
        l1 = l2 * 3 / 4
    l1, l2 = l1 or 3.0, l2 or 4.0
    P = {r: (0.0, 0.0), p: (l1, 0.0), q: (0.0, l2)}
    return {x: P[x] for x in (A, B, C)}


def b_right_isosceles(n, S, data):
    (A, B, C), r, p, q = split_right(n, data)
    leg = S(r, p) or S(r, q) or ((S(p, q) or 0) / math.sqrt(2)) or 4.0
    P = {r: (0.0, 0.0), p: (leg, 0.0), q: (0.0, leg)}
    return {x: P[x] for x in (A, B, C)}


def b_isosceles(n, S, data):
    A, B, C = n[:3]
    a = str(data.get("apex") or A)
    if a not in (A, B, C):
        raise ValueError("apex không nằm trong names")
    p, q = [x for x in (A, B, C) if x != a]
    base = S(p, q) or 4.0
    h = data.get("height")
    if h is None:
        leg = S(a, p) or S(a, q)
        h = root(leg * leg - base * base / 4, "cạnh bên quá ngắn") if leg else 0.9 * base
    P = {p: (-base / 2, 0.0), q: (base / 2, 0.0), a: (0.0, float(h))}
    return {x: P[x] for x in (A, B, C)}


def b_equilateral(n, S, data):
    A, B, C = n[:3]
    s = float(data.get("side") or S(A, B) or S(B, C) or S(A, C) or 4.0)
    return {A: (0.0, 0.0), B: (s, 0.0), C: (s / 2, s * math.sqrt(3) / 2)}


def b_rectangle(n, S, data):
    A, B, C, D = n[:4]
    w, h = S(A, B) or S(C, D), S(A, D) or S(B, C)
    if w and not h:
        h = 0.6 * w
    elif h and not w:
        w = h / 0.6
    w, h = w or 4.0, h or 2.4
    return {A: (0.0, 0.0), B: (w, 0.0), C: (w, h), D: (0.0, h)}


def b_square(n, S, data):
    A, B, C, D = n[:4]
    s = float(data.get("side") or S(A, B) or S(B, C) or S(C, D) or S(A, D) or 3.0)
    return {A: (0.0, 0.0), B: (s, 0.0), C: (s, s), D: (0.0, s)}


def _slanted(n, ab, ad, data):
    A, B, C, D = n[:4]
    th = math.radians(float(data.get("angle") or 60))
    dx, dy = ad * math.cos(th), ad * math.sin(th)
    return {A: (0.0, 0.0), B: (ab, 0.0), C: (ab + dx, dy), D: (dx, dy)}


def b_parallelogram(n, S, data):
    A, B, C, D = n[:4]
    ab, ad = S(A, B) or S(C, D), S(A, D) or S(B, C)
    if ab and not ad:
        ad = 0.6 * ab
    elif ad and not ab:
        ab = ad / 0.6
    return _slanted(n, ab or 5.0, ad or 3.0, data)


def b_rhombus(n, S, data):
    A, B, C, D = n[:4]
    s = float(data.get("side") or S(A, B) or S(B, C) or S(C, D) or S(A, D) or 4.0)
    return _slanted(n, s, s, data)


def _bases(A, B, C, D, S, ab_default, cd_default):
    ab, cd = S(A, B), S(C, D)
    if ab and not cd:
        cd = ab * cd_default / ab_default
    elif cd and not ab:
        ab = cd * ab_default / cd_default
    return ab or ab_default, cd or cd_default


def b_trapezoid(n, S, data):
    A, B, C, D = n[:4]
    ab, cd = _bases(A, B, C, D, S, 6.0, 3.0)
    h = float(data.get("height") or 0.5 * max(ab, cd))
    dx = 0.15 * ab
    return {A: (0.0, 0.0), B: (ab, 0.0), C: (dx + cd, h), D: (dx, h)}


def b_isosceles_trapezoid(n, S, data):
    A, B, C, D = n[:4]
    ab, cd = _bases(A, B, C, D, S, 6.0, 3.0)
    dx = (ab - cd) / 2
    h = data.get("height")
    if h is None:
        leg = S(A, D) or S(B, C)
        h = root(leg * leg - dx * dx, "cạnh bên quá ngắn") if leg else 0.5 * max(ab, cd)
    h = float(h)
    return {A: (0.0, 0.0), B: (ab, 0.0), C: (dx + cd, h), D: (dx, h)}


def b_right_trapezoid(n, S, data):
    # vuông tại A và D: đáy AB, CD; đường cao AD
    A, B, C, D = n[:4]
    ab, cd = _bases(A, B, C, D, S, 4.0, 6.0)
    ad = S(A, D)
    if ad is None and S(A, B) and S(C, D) and S(B, C):
        ad = root(S(B, C) ** 2 - (cd - ab) ** 2, "BC quá ngắn")
    ad = ad or 0.6 * max(ab, cd)
    return {A: (0.0, 0.0), B: (ab, 0.0), C: (cd, ad), D: (0.0, ad)}


def b_quadrilateral(n, S, data):
    A, B, C, D = n[:4]
    return fit({A: (0.0, 0.0), B: (5.0, 0.0), C: (4.0, 3.5), D: (0.8, 2.8)}, S)


def b_regular_polygon(n, S, data):
    k = len(n)
    if k < 3:
        raise ValueError("Đa giác đều cần ít nhất 3 đỉnh")
    s = float(data.get("side") or S(n[0], n[1]) or 3.0)
    R = s / (2 * math.sin(math.pi / k))
    a0 = -math.pi / 2 - math.pi / k  # đỉnh đầu ở góc dưới bên trái, cạnh đáy nằm ngang
    return {nm: (R * math.cos(a0 + 2 * math.pi * i / k), R * math.sin(a0 + 2 * math.pi * i / k))
            for i, nm in enumerate(n)}


BUILDERS = {
    "triangle": b_triangle,
    "right_triangle": b_right_triangle,
    "right_isosceles_triangle": b_right_isosceles,
    "isosceles_triangle": b_isosceles,
    "equilateral_triangle": b_equilateral,
    "rectangle": b_rectangle,
    "square": b_square,
    "parallelogram": b_parallelogram,
    "rhombus": b_rhombus,
    "trapezoid": b_trapezoid,
    "isosceles_trapezoid": b_isosceles_trapezoid,
    "right_trapezoid": b_right_trapezoid,
    "quadrilateral": b_quadrilateral,
    "regular_polygon": b_regular_polygon,
}


def _need_name(name):
    if not name:
        raise ValueError("Thiếu tên điểm")
    return name


def build_figure(data: dict) -> dict:
    fig = data.get("figure")
    if fig not in ("circle", "none") and fig not in BUILDERS:
        raise ValueError("Loại hình chưa hỗ trợ: " + str(fig))

    names = [str(c) for c in (data.get("names") or "") if str(c).strip()]
    sides = get_sides(data)
    S = lambda p, q: sides.get(p + q) or sides.get(q + p)
    shapes, segs, texts = [], [], []
    lines, drawn, counters = {}, [], {}  # lines: đường thẳng có tên; drawn: các đường sẽ vẽ

    try:
        if fig == "circle":
            o = str(data.get("center") or "O")
            try:
                r = float(data.get("r") or data.get("radius"))
            except (TypeError, ValueError):  # không có số (R, a...) -> cỡ mặc định, không ghi gì
                r = 3.0
            if r <= 0:
                r = 3.0
            pts = {o: (0.0, 0.0)}
            shapes.append({"type": "circle", "cx": 0.0, "cy": 0.0, "r": r,
                           "color": "tab:blue", "fill": False})
            # điểm thuộc đường tròn: đặt ở các góc cố định, không nối bán kính
            on = [str(x) for x in list(data.get("on_circle") or [])]
            k = len(on)
            angs = {1: [90], 2: [150, 30], 3: [150, 30, 270]}.get(k) or \
                   [150 - i * 360 / k for i in range(k)]
            for nm, a in zip(on, angs):
                pts[nm] = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        elif fig == "none":
            pts = {}
        else:
            pts = BUILDERS[fig](names, S, data)
            segs.append(list(pts.values()) + [next(iter(pts.values()))])

            # ghi số đo các cạnh mà đề cho, đặt lệch ra phía ngoài hình
            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            size = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            for key, v in sides.items():
                p, q = key[0], key[1]
                if p in pts and q in pts and p != q:
                    mx, my = mid(pts[p], pts[q])
                    dx, dy = mx - cx, my - cy
                    d = math.hypot(dx, dy) or 1.0
                    texts.append({"x": mx + dx / d * 0.07 * size,
                                  "y": my + dy / d * 0.07 * size, "text": f"{v:g}"})

        def extent():
            if not pts:
                return 0.0
            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            return max(max(xs) - min(xs), max(ys) - min(ys))

        def make_line(P, Q):
            L = math.dist(P, Q)
            if L < 1e-9:
                raise ValueError("Hai điểm trùng nhau")
            return {"p": mid(P, Q), "v": ((Q[0] - P[0]) / L, (Q[1] - P[1]) / L),
                    "scale": L, "kind": "through"}

        def get_line(ref):
            """ref = tên đường thẳng ("d") hoặc 2 điểm ("BC" / ["B","C"])."""
            if isinstance(ref, str) and ref in lines:
                return lines[ref]
            ns = list(ref)
            if len(ns) == 2 and all(x in pts for x in ns):
                return make_line(pts[ns[0]], pts[ns[1]])
            raise ValueError("Không tìm thấy đường thẳng " + str(ref))

        for e in data.get("extras") or []:
            t, name = e.get("type"), str(e.get("name") or "")
            of = list(e.get("of") or "")
            if t == "midpoint":
                pts[_need_name(name)] = mid(pts[of[0]], pts[of[1]])
            elif t == "foot":
                frm, ln = str(e["from"]), get_line(e["to"])
                pts[_need_name(name)] = foot(
                    pts[frm], ln["p"], (ln["p"][0] + ln["v"][0], ln["p"][1] + ln["v"][1]))
                segs.append([pts[frm], pts[name]])
            elif t == "on_segment":
                p, q = pts[of[0]], pts[of[1]]
                k = float(e.get("ratio") or 0.4)
                pts[_need_name(name)] = (p[0] + k * (q[0] - p[0]), p[1] + k * (q[1] - p[1]))
            elif t == "centroid":
                P = [pts[x] for x in of[:3]]
                pts[_need_name(name)] = (sum(p[0] for p in P) / 3, sum(p[1] for p in P) / 3)
            elif t == "circumcircle":
                name = name or "O"
                P = [pts[x] for x in of[:3]]
                o = circumcenter(*P)
                pts[name] = o
                shapes.append({"type": "circle", "cx": o[0], "cy": o[1],
                               "r": math.dist(o, P[0]), "color": "tab:green", "fill": False})
            elif t == "line":
                th = e.get("through")
                tn = list(th) if isinstance(th, (str, list)) else []
                if e.get("tangent_at"):
                    a, o = str(e["tangent_at"]), str(e.get("of") or "")
                    rx, ry = pts[a][0] - pts[o][0], pts[a][1] - pts[o][1]
                    r0 = math.hypot(rx, ry) or 1.0
                    ln = {"p": pts[a], "v": (-ry / r0, rx / r0), "scale": r0, "kind": "free"}
                elif e.get("perpendicular_to") is not None or e.get("parallel_to") is not None:
                    perp = e.get("perpendicular_to") is not None
                    ref = get_line(e["perpendicular_to"] if perp else e["parallel_to"])
                    v = nrm(ref["v"]) if perp else ref["v"]
                    if tn:
                        p = pts[tn[0]]
                    elif perp:
                        p = ref["p"]
                    else:  # song song mà không nói qua điểm nào: dịch đường kia sang một đoạn
                        nv = nrm(ref["v"])
                        p = (ref["p"][0] + nv[0] * 0.6 * ref["scale"],
                             ref["p"][1] + nv[1] * 0.6 * ref["scale"])
                    ln = {"p": p, "v": v, "scale": ref["scale"], "kind": "free"}
                elif len(tn) >= 2:
                    ln = make_line(pts[tn[0]], pts[tn[1]])
                elif len(tn) == 1:
                    ln = {"p": pts[tn[0]], "v": (1.0, 0.0),
                          "scale": max(3.0, extent()), "kind": "free"}
                else:  # đường thẳng tự do: đường đầu nằm ngang, đường sau xoay dần
                    k = counters.get("free", 0)
                    counters["free"] = k + 1
                    ang = math.radians([0, 60, 120, 30, 90][k % 5])
                    ln = {"p": (0.0, 0.0), "v": (math.cos(ang), math.sin(ang)),
                          "scale": max(4.0 if not pts else extent(), 3.0), "kind": "free"}
                if name:
                    lines[name] = ln
                drawn.append({**ln, "name": name})
            elif t in ("on_line", "off_line"):
                ref = e.get("of")
                ln = get_line(ref)
                key = (t, str(ref))
                k = counters.get(key, 0)
                counters[key] = k + 1
                if t == "on_line":
                    seq = ON_THROUGH if ln["kind"] == "through" else ON_FREE
                    s, off = seq[k % len(seq)] * ln["scale"], 0.0
                else:
                    s, off = OFF_SEQ[k % len(OFF_SEQ)] * ln["scale"], 0.55 * ln["scale"]
                nv = nrm(ln["v"])
                pts[_need_name(name)] = (ln["p"][0] + ln["v"][0] * s + nv[0] * off,
                                         ln["p"][1] + ln["v"][1] * s + nv[1] * off)
            elif t == "intersection":
                l1, l2 = get_line(e["of"][0]), get_line(e["of"][1])
                den = l1["v"][0] * l2["v"][1] - l1["v"][1] * l2["v"][0]
                if abs(den) < 1e-9:
                    raise ValueError("Hai đường thẳng song song, không có giao điểm")
                dx, dy = l2["p"][0] - l1["p"][0], l2["p"][1] - l1["p"][1]
                tt = (dx * l2["v"][1] - dy * l2["v"][0]) / den
                pts[_need_name(name)] = (l1["p"][0] + l1["v"][0] * tt,
                                         l1["p"][1] + l1["v"][1] * tt)
            else:
                raise ValueError("extras chưa hỗ trợ: " + str(t))

        for s in data.get("draw") or []:
            chain = [pts[x] for x in list(s) if x in pts]
            if len(chain) >= 2:
                segs.append(chain)
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("Model trả thiếu/sai dữ liệu: " + repr(e))

    return {"points": pts, "segments": segs, "texts": texts, "shapes": shapes, "lines": drawn}


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

def clip_line(p, v, x0, x1, y0, y1):
    """Cắt đường thẳng (qua p, hướng v) theo hình chữ nhật khung nhìn."""
    lo, hi = -1e9, 1e9
    for pc, vc, a, b in ((p[0], v[0], x0, x1), (p[1], v[1], y0, y1)):
        if abs(vc) < 1e-12:
            if not (a <= pc <= b):
                return None
        else:
            t1, t2 = (a - pc) / vc, (b - pc) / vc
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    if lo >= hi:
        return None
    return (p[0] + v[0] * lo, p[1] + v[1] * lo), (p[0] + v[0] * hi, p[1] + v[1] * hi)


def draw_line(ax, L, box):
    seg = clip_line(L["p"], L["v"], *box)
    if not seg:
        return
    (xa, ya), (xb, yb) = seg
    if (xb, yb) < (xa, ya):  # đầu mút "lớn hơn" ở bên phải/trên, để ghi tên ở đó
        xa, ya, xb, yb = xb, yb, xa, ya
    ax.plot([xa, xb], [ya, yb], color="tab:purple", lw=1.8)
    if L.get("name"):
        nx, ny = -L["v"][1], L["v"][0]
        if ny < 0 or (abs(ny) < 1e-9 and nx < 0):
            nx, ny = -nx, -ny
        px, py = xa + (xb - xa) * 0.93, ya + (yb - ya) * 0.93
        ax.annotate(L["name"], (px, py), xytext=(nx * 12, ny * 12), textcoords="offset points",
                    ha="center", va="center", fontsize=13, fontstyle="italic",
                    color="tab:purple")


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

    # chốt khung nhìn trước (theo điểm/hình/đoạn đại diện của đường thẳng), rồi mới vẽ
    # đường thẳng kéo dài tới mép khung
    for L in d.get("lines", []):
        hh = 0.5 * L.get("scale", 4.0)
        ax.plot([L["p"][0] - L["v"][0] * hh, L["p"][0] + L["v"][0] * hh],
                [L["p"][1] - L["v"][1] * hh, L["p"][1] + L["v"][1] * hh], alpha=0)
    if d["points"] or d["shapes"] or d["segments"] or d["texts"] or d.get("lines"):
        ax.autoscale_view()
        ax.margins(0.2)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
    else:
        x0, x1, y0, y1 = -5.0, 5.0, -3.0, 3.0
    w, h = x1 - x0, y1 - y0
    if h < 0.5 * w:  # khung quá dẹt (vd. mọi điểm nằm trên 1 đường ngang) -> nới ra
        c = (y0 + y1) / 2
        y0, y1 = c - 0.25 * w, c + 0.25 * w
    elif w < 0.5 * h:
        c = (x0 + x1) / 2
        x0, x1 = c - 0.25 * h, c + 0.25 * h
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for L in d.get("lines", []):
        draw_line(ax, L, (x0, x1, y0, y1))
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
            p = input("Đề (q để thoát): ").strip()
            if p.lower() in ("q", "quit", ""):
                break
            run(p)