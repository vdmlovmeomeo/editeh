import json
import math
import re
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.patches import Arc, Circle, Polygon
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepscaler"
MAX_RETRY = 3
N_EXAMPLES = 3  # số ví dụ giống nhất đưa vào prompt
EXAMPLES_PATH = Path(__file__).with_name("examples.jsonl")
OUTPUT_DIR = Path(__file__).with_name("outputs")

PROMPT_HEAD = """You convert a geometry problem (Vietnamese or English) into JSON.
DO NOT compute coordinates and NEVER invent numbers. Only extract the figure type, the point names and the lengths the problem actually gives. A program does the geometry and draws a default shape when no length is given.

Output ONLY one JSON object. Fields:
- "figure": triangle | right_triangle | right_isosceles_triangle | isosceles_triangle | equilateral_triangle | rectangle | square | parallelogram | rhombus | trapezoid | isosceles_trapezoid | right_trapezoid | quadrilateral | kite | regular_polygon | circle | semicircle | two_circles | none (only lines and points) | coordinate | tetrahedron | pyramid | prism | cube
- "names": vertex names in order, e.g. "ABC", "ABCD", "ABCDEF"
- given lengths as two-letter keys, ONLY if the problem gives numbers, e.g. "AB": 3, "AC": 4
- "right_at": vertex with the right angle (right_triangle, right_isosceles_triangle)
- "apex" (and "height" if given) for isosceles_triangle
- "side": side length (equilateral_triangle, square, rhombus, regular_polygon), only if given
- "angle": angle at the first vertex in degrees (parallelogram, rhombus), only if given
- trapezoid / isosceles_trapezoid / right_trapezoid: the bases are AB and CD (AB // CD); right_trapezoid has right angles at A and D and AD is the height; "height" only if given
- regular_polygon: "names" has one letter per vertex
- circle: "center"; "r" ONLY if a NUMBER is given (never put a letter in r). "(O;R)" means center O and radius R.
- "on_circle": names of the points lying on the circle, e.g. ["A","B"]; a circle may also have "diameter":["A","B"]
- semicircle: "center","names":"AB","on_circle":[...]. two_circles: "centers":["O","O'"], "relation": intersect|tangent_external|tangent_internal|outside|inside, "common":["A","B"], "r1"/"r2" only if given
- "angle_A":60 = angle of the triangle at A in degrees (only if given); "kind":"obtuse" for an obtuse triangle; kite: AB=AD, CB=CD
- coordinate (Oxy): "points":{"A":[1,2]}, "graphs":[{"kind":"linear","a":2,"b":1,"name":"d"},{"kind":"quadratic","a":1,"b":0,"c":0,"name":"P"}]
- 3D: tetrahedron "ABCD"; pyramid "names":"S.ABCD" (+ "base":"square", "apex_over":"A"); prism/cube "names":"ABCD.A'B'C'D'". 3D lengths like "AA'":5, "SA":4
- "extras": extra points, each one of
  {"name":"M","type":"midpoint","of":"BC"}
  {"name":"H","type":"foot","from":"A","to":"BC"}     (foot of the perpendicular from A to line BC, or to a named line d)
  {"name":"N","type":"on_segment","of":"BC"}         (any point lying on segment BC)
  {"name":"G","type":"centroid","of":"ABC"}
  {"name":"I","type":"incenter","of":"ABC"}       (tâm đường tròn nội tiếp tam giác)
  {"name":"H","type":"orthocenter","of":"ABC"}     (trực tâm tam giác)
  {"name":"O","type":"circumcircle","of":"ABC"}       (circumcenter + circumscribed circle)
  {"type":"line","name":"d","through":["A","B"]}       (line through two points)
  {"type":"line","name":"d","through":"M","parallel_to":"BC"}       (parallel_to / perpendicular_to take a line name or two point names)
  {"type":"line","name":"d","through":"M","perpendicular_to":"BC"}
  {"type":"line","name":"t","tangent_at":"A","of":"O"}    (tangent at A to the circle with center O)
  {"type":"line","name":"d"}                               (a free line, no "through")
  {"name":"A","type":"on_line","of":"d"}                   (point lying on line d)
  {"name":"M","type":"off_line","of":"d"}                  (point NOT lying on line d)
  {"name":"M","type":"intersection","of":["d","BC"]}       (intersection of two lines)
  {"name":"D","type":"bisector_foot","from":"A","of":"BC"}     (angle bisector from A meets BC at D)
  {"name":"H","type":"orthocenter","of":"ABC","feet":["D","E","F"]}   (feet of the altitudes are optional)
  {"name":"I","type":"incircle","of":"ABC","touch":["D","E","F"]}      (incenter + incircle; touch points optional)
  {"name":"D","type":"reflect","point":"A","over":"O"}      (symmetric of A over point O, or over a line like "BC")
  {"name":"D","type":"parallelogram_vertex","of":"ABC"}     (ABCD is a parallelogram)
  {"name":"E","type":"extend","from":"A","through":"B","ratio":1.5}    (E on ray AB beyond B)
  {"name":"O","type":"center_of","of":"ABCD"}
  {"name":"N","type":"on_segment","of":"CD","angle":{"vertex":"A","other":"M","value":45}}   (N on CD such that angle MAN = 45)
  {"name":"N","type":"on_segment","of":"BC","dist":2}     (BN = 2)
  {"type":"line","name":"d","perp_bisector_of":"AB"}   {"type":"ray","name":"x","from":"A","through":"B"}
  {"type":"circle","center":"O","through":"A"}   or "diameter":["A","B"]   or "r":2     (an extra circle)
  {"name":"D","type":"on_circle","of":"O"}               (point on the circle with center O)
  {"type":"tangents","from":"M","of":"O","names":["A","B"]}   (two tangents from M)
  {"type":"secant","from":"M","of":"O","names":["A","B"]}
  {"type":"line_circle","line":"d","of":"O","names":["A","B"]}
  {"type":"graph_intersection","of":["P","d"],"names":["A","B"]}
  extras are built in order: list a point before any line that uses it.
- "draw": segments to draw ONLY if the problem asks (nối, dây, bán kính, đường kính...), e.g. ["AM"], ["OA"]
- also: "dashed":["AD"], "vectors":[["A","B"]], "fill":[["A","B","C"]], "side_labels":{"AB":"a"},
  "marks":[{"type":"right_angle","at":"A","between":["B","C"]},{"type":"equal_sides","sides":["AB","AC"],"ticks":1},{"type":"angle","at":"A","between":["B","C"],"label":"60°"},{"type":"equal_angles","angles":[["B","A","M"],["M","A","C"]]}]

IMPORTANT: Never omit a stated construction. If two lines intersect, a point lies on a side,
or segments are connected, encode every fact using "extras" and/or "draw", even if the base figure is clear.
For "M thuộc BC; AM cắt BD tại E; nối FM", define M as on_segment, then E as the
intersection of AM and BD, and include AM/FM in draw."""

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


CASES_PATH = Path(__file__).with_name("train_cases.txt")


def load_cases():
    """Đọc train_cases.txt: mỗi dòng "đề ||| {json}"; dòng bắt đầu bằng # là tên nhóm/ghi chú."""
    out = []
    if not CASES_PATH.exists():
        return out
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|||" not in line:
            continue
        prob, js = line.split("|||", 1)
        try:
            out.append((prob.strip(), json.loads(js.strip())))
        except ValueError:
            continue
    return out


_have = {p for p, _ in SEED_EXAMPLES}
SEED_EXAMPLES = SEED_EXAMPLES + [c for c in load_cases() if c[0] not in _have]


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
        "options": {"temperature": temp, "num_ctx": 8192, "num_predict": 1000},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        detail = r.text.strip() or "(Ollama không trả nội dung lỗi)"
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {detail}") from e
    return r.json()["response"]


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("Không tìm thấy JSON")
    return json.loads(text[a : b + 1])


def remove_invented_measurements(data: dict, prompt: str) -> dict:
    """Không cho model tự thêm số đo cạnh/góc không có trong đề."""
    if not data.get("figure"):
        return data
    raw_numbers = re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", prompt)
    allowed = {x.replace(",", ".") for x in raw_numbers}
    out = dict(data)
    for key, value in list(out.items()):
        is_measurement = key in {"side", "angle", "height", "r", "r1", "r2"} or str(key).startswith("angle_") or (
            isinstance(key, str) and re.fullmatch(r"[A-Z]{2}", key)
        )
        if is_measurement and isinstance(value, (int, float)):
            if str(value) not in allowed and f"{value:g}" not in allowed:
                out.pop(key)
    return out


def enforce_explicit_figure(data: dict, prompt: str) -> dict:
    """Ưu tiên loại hình được nói rõ trong đề (cụm được nhắc đến ĐẦU TIÊN) hơn nhãn model đoán."""
    p = prompt.lower()
    out = dict(data)
    nm = r"(?:[a-z]{3,6}\s+)?"  # cho phép tên đỉnh xen giữa: "tam giác abc vuông"
    explicit = (
        (r"hình lập phương", "cube"), (r"hình hộp", "prism"), (r"lăng trụ", "prism"),
        (r"hình chóp", "pyramid"), (r"tứ diện", "tetrahedron"),
        (r"nửa đường tròn", "semicircle"), (r"hai đường tròn", "two_circles"),
        (r"hình diều", "kite"),
        (r"hình vuông", "square"), (r"hình chữ nhật", "rectangle"),
        (r"hình bình hành", "parallelogram"), (r"hình thoi", "rhombus"),
        (r"hình thang\s+" + nm + r"cân", "isosceles_trapezoid"),
        (r"hình thang\s+" + nm + r"vuông", "right_trapezoid"),
        (r"hình thang", "trapezoid"),
        (r"tam giác\s+" + nm + r"đều", "equilateral_triangle"),
        (r"tam giác\s+" + nm + r"vuông\s+cân", "right_isosceles_triangle"),
        (r"tam giác\s+" + nm + r"vuông", "right_triangle"),
        (r"tam giác\s+" + nm + r"cân", "isosceles_triangle"),
        (r"tam giác", "triangle"),
    )
    if any(w in p for w in ("oxy", "hệ trục", "mặt phẳng toạ độ", "mặt phẳng tọa độ")):
        found = {}
        for mm in re.finditer(r"\b([A-Z])\s*\(\s*(-?\d+(?:[.,]\d+)?)\s*[;,]\s*(-?\d+(?:[.,]\d+)?)\s*\)", prompt):
            found[mm.group(1)] = [float(mm.group(2).replace(",", ".")), float(mm.group(3).replace(",", "."))]
        if found:  # toạ độ ghi trong đề đáng tin hơn những gì model chép lại
            out["figure"] = "coordinate"
            out["points"] = found
            out.pop("names", None)
    if out.get("figure") != "coordinate" and not any(w in p for w in ("oxy", "đồ thị", "hệ trục")):
        best = None
        for i, (pat, figure) in enumerate(explicit):
            m0 = re.search(pat, p)
            if m0 and (best is None or (m0.start(), i) < best[0]):
                best = ((m0.start(), i), figure)
        if best:
            out["figure"] = best[1]
    f3 = out.get("figure")
    if f3 == "pyramid":
        m3 = re.search(r"\b([A-Z])\.([A-Z]{3,4})\b", prompt)
        if m3:
            out["names"] = m3.group(1) + "." + m3.group(2)
    elif f3 in ("prism", "cube"):
        m3 = re.search(r"\b([A-Z]{3,4})\.((?:[A-Z]['’′]){3,4})", prompt)
        if m3:
            out["names"] = m3.group(1) + "." + m3.group(2)
    elif f3 == "tetrahedron":
        m3 = re.search(r"tứ diện\s+([A-Z]{4})", prompt, re.I)
        if m3:
            out["names"] = m3.group(1).upper()
    # Tên đỉnh nêu rõ trong đề đáng tin hơn tên model đoán (chỉ áp dụng cho tứ giác).
    quad = {"square", "rectangle", "parallelogram", "rhombus", "trapezoid",
            "isosceles_trapezoid", "right_trapezoid", "kite", "quadrilateral"}
    m = re.search(r"(?:hình vuông|hình chữ nhật|hình bình hành|hình thoi|hình diều|"
                  r"hình thang(?: cân| vuông)?)\s+([A-Z]{4})", prompt, re.I)
    if m and out.get("figure") in quad:
        out["names"] = m.group(1).upper()
    # Không cho điểm phụ ghi đè lên các đỉnh nền (ví dụ M/N/P/Q của hình bình hành).
    base_names = set(str(out.get("names") or ""))
    if base_names and out.get("extras"):
        out["extras"] = [e for e in out["extras"]
                         if str(e.get("name") or "") not in base_names]
    return out


def enforce_diagonal_construction(data: dict, prompt: str) -> dict:
    """Recover the common 'diagonals intersect at O, connect O to vertices' form."""
    out = dict(data)
    extras = list(out.get("extras") or [])
    draws = list(out.get("draw") or [])
    m = re.search(r"đường chéo\s+([A-Z]{2})\s+và\s+([A-Z]{2})\s+cắt nhau tại\s+([A-Z])", prompt, re.I)
    if m:
        a, b, o = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
        if not any(e.get("name") == o and e.get("type") == "intersection" for e in extras):
            extras.append({"name": o, "type": "intersection", "of": [a, b]})
        for seg in (a, b):
            if seg not in draws:
                draws.append(seg)
        n = re.search(r"nối\s+" + re.escape(o) + r"\s+với\s+([A-Z](?:\s*,\s*[A-Z])*)", prompt, re.I)
        if n:
            for v in re.findall(r"[A-Z]", n.group(1).upper()):
                seg = o + v
                if seg not in draws:
                    draws.append(seg)
    out["extras"], out["draw"] = extras, draws
    return out


def normalize_prompt(prompt: str) -> str:
    """Chuẩn hoá ký hiệu, viết tắt và vài cách đảo câu hình học."""
    p = str(prompt)
    replacements = {
        "△": " tam giác ", "Δ": " tam giác ", "∠": " góc ",
        "⊥": " vuông góc ", "⟂": " vuông góc ", "∥": " song song ",
        "||": " song song ", "°": " độ ",
    }
    for old, new in replacements.items():
        p = p.replace(old, new)
    # Viết tắt phổ biến, chỉ thay khi là một từ riêng.
    p = re.sub(r"\b(?:tg|tgiac)\b", "tam giác", p)
    p = re.sub(r"\bhv\b", "hình vuông", p)
    p = re.sub(r"\bhcn\b", "hình chữ nhật", p)
    p = re.sub(r"\bbh\b", "hình bình hành", p)
    p = re.sub(r"\bđt\b", "đường thẳng", p)
    # ABCD là hình vuông / ABC là tam giác ABC -> hình vuông ABCD.
    p = re.sub(r"\b([A-Z]{3,6})\s+là\s+(hình\s+(?:vuông|chữ nhật|bình hành|thoi|thang)|tam giác(?:\s+đều|\s+cân|\s+vuông)?)",
               r"\2 \1", p, flags=re.I)
    return re.sub(r"\s+", " ", p).strip()


def enforce_explicit_relations(data: dict, prompt: str) -> dict:
    """Tự vẽ các đoạn/đường xuất hiện trong quan hệ vuông góc/song song."""
    out = dict(data)
    draws = list(out.get("draw") or [])
    # AC vuông góc BD / AC song song BD (đã được normalize về chữ).
    for pattern in (r"\b([A-Z]{2})\s+vuông góc\s+([A-Z]{2})\b",
                    r"\b([A-Z]{2})\s+song song\s+([A-Z]{2})\b"):
        for a, b in re.findall(pattern, prompt, flags=re.I):
            for seg in (a.upper(), b.upper()):
                if seg not in draws:
                    draws.append(seg)
    out["draw"] = draws
    return out


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


def incenter(a, b, c):
    """Tâm nội tiếp: trọng số theo độ dài cạnh đối diện."""
    wa, wb, wc = math.dist(b, c), math.dist(a, c), math.dist(a, b)
    s = wa + wb + wc
    if s < 1e-12:
        raise ValueError("Tam giác suy biến")
    return ((wa * a[0] + wb * b[0] + wc * c[0]) / s,
            (wa * a[1] + wb * b[1] + wc * c[1]) / s)


def orthocenter(a, b, c):
    """Trực tâm là giao của đường cao qua A và đường cao qua B."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    # A + t*n(BC), B + u*n(AC)
    n1 = (- (c[1] - b[1]), c[0] - b[0])
    n2 = (- (c[1] - a[1]), c[0] - a[0])
    den = n1[0] * n2[1] - n1[1] * n2[0]
    if abs(den) < 1e-12:
        raise ValueError("Tam giác suy biến")
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = (dx * n2[1] - dy * n2[0]) / den
    return (a[0] + t * n1[0], a[1] + t * n1[1])


def root(x, what="số liệu"):
    if x <= 1e-9:
        raise ValueError("Không hợp lệ: " + what)
    return math.sqrt(x)


def get_sides(data: dict) -> dict:
    src = {**data, **(data.get("sides") or {})}
    out = {}
    for k, v in src.items():
        if isinstance(k, str) and re.fullmatch(r"[A-Z]'?[A-Z]'?", k):
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

def _b_triangle_old(n, S, data):
    A, B, C = n[:3]
    c, b, a = S(A, B), S(A, C), S(B, C)
    if a and b and c:
        x = (b * b + c * c - a * a) / (2 * c)
        y = root(b * b - x * x, "3 cạnh không tạo thành tam giác")
        return {A: (0.0, 0.0), B: (c, 0.0), C: (x, y)}
    return fit({A: (0.0, 0.0), B: (5.0, 0.0), C: (1.8, 3.6)}, S)


def _b_right_triangle_old(n, S, data):
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


def _b_isosceles_old(n, S, data):
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


# ---------------------------------------------------------------- v6: tam giác biết góc, hình diều

def tri_angles(data, n):
    """Các góc đề cho: "angle_A": 60 -> {"A": 60.0}."""
    out = {}
    for x in n[:3]:
        v = data.get("angle_" + x)
        if v is None:
            continue
        try:
            out[x] = float(v)
        except (TypeError, ValueError):
            continue
        if not 0 < out[x] < 180:
            raise ValueError("Góc phải nằm trong khoảng (0, 180)")
    return out


def _from_sss(A, B, C, a, b, c):
    x = (b * b + c * c - a * a) / (2 * c)
    y = root(b * b - x * x, "3 cạnh không tạo thành tam giác")
    return {A: (0.0, 0.0), B: (c, 0.0), C: (x, y)}


def b_triangle(n, S, data):
    A, B, C = n[:3]
    ang = tri_angles(data, n)
    if len(ang) >= 2:  # biết 2 (hoặc 3) góc -> góc còn lại; cạnh co giãn theo cạnh đề cho
        if len(ang) == 2:
            m = [x for x in (A, B, C) if x not in ang][0]
            ang[m] = 180.0 - sum(ang.values())
            if ang[m] <= 0:
                raise ValueError("Tổng hai góc phải nhỏ hơn 180")
        k = {x: math.sin(math.radians(ang[x])) for x in (A, B, C)}
        opp = {A: S(B, C), B: S(A, C), C: S(A, B)}
        s0 = next((opp[x] / k[x] for x in (A, B, C) if opp[x]), 5.0 / k[C])
        return _from_sss(A, B, C, s0 * k[A], s0 * k[B], s0 * k[C])
    if len(ang) == 1:  # 1 góc + cạnh kề (thiếu thì lấy cỡ mặc định)
        X = next(iter(ang))
        Y, Z = [x for x in (A, B, C) if x != X]
        l1, l2 = S(X, Y), S(X, Z)
        l1 = l1 or (l2 * 1.25 if l2 else 5.0)
        l2 = l2 or l1 * 0.8
        th = math.radians(ang[X])
        P = {X: (0.0, 0.0), Y: (l1, 0.0), Z: (l2 * math.cos(th), l2 * math.sin(th))}
        return {x: P[x] for x in (A, B, C)}
    if not (S(A, B) and S(A, C) and S(B, C)) and str(data.get("kind") or "") == "obtuse":
        return fit({A: (0.0, 0.0), B: (5.0, 0.0), C: (-1.5, 2.5)}, S)
    return _b_triangle_old(n, S, data)


def b_right_triangle(n, S, data):
    (A, B, C), r, p, q = split_right(n, data)
    ang = tri_angles(data, n)
    tp, tq = ang.get(p), ang.get(q)
    if tp is None and tq is None:
        return _b_right_triangle_old(n, S, data)
    l1, l2 = S(r, p), S(r, q)
    if tp is not None:
        if not 0 < tp < 90:
            raise ValueError("Góc nhọn phải nằm trong (0, 90)")
        ratio = math.tan(math.radians(tp))  # tan(góc tại p) = l2 / l1
        if l1:
            l2 = l1 * ratio
        elif l2:
            l1 = l2 / ratio
        else:
            l1 = 4.0
            l2 = l1 * ratio
    else:
        if not 0 < tq < 90:
            raise ValueError("Góc nhọn phải nằm trong (0, 90)")
        ratio = math.tan(math.radians(tq))  # tan(góc tại q) = l1 / l2
        if l2:
            l1 = l2 * ratio
        elif l1:
            l2 = l1 / ratio
        else:
            l2 = 4.0
            l1 = l2 * ratio
    P = {r: (0.0, 0.0), p: (l1, 0.0), q: (0.0, l2)}
    return {x: P[x] for x in (A, B, C)}


def b_isosceles(n, S, data):
    A, B, C = n[:3]
    a = str(data.get("apex") or A)
    if a not in (A, B, C):
        raise ValueError("apex không nằm trong names")
    p, q = [x for x in (A, B, C) if x != a]
    ang = tri_angles(data, n)
    if not ang:
        return _b_isosceles_old(n, S, data)
    base = S(p, q)
    if a in ang:  # biết góc ở đỉnh
        half = math.radians(ang[a]) / 2
        if base:
            h = (base / 2) / math.tan(half)
        else:
            leg = S(a, p) or S(a, q) or 5.0
            base, h = 2 * leg * math.sin(half), leg * math.cos(half)
    else:  # biết góc ở đáy
        be = ang.get(p) or ang.get(q)
        if not 0 < be < 90:
            raise ValueError("Góc ở đáy phải nằm trong (0, 90)")
        base = base or 4.0
        h = (base / 2) * math.tan(math.radians(be))
    P = {p: (-base / 2, 0.0), q: (base / 2, 0.0), a: (0.0, float(h))}
    return {x: P[x] for x in (A, B, C)}


def b_kite(n, S, data):
    # hình diều: AB = AD, CB = CD, đối xứng qua đường chéo AC
    A, B, C, D = n[:4]
    return fit({A: (-2.0, 0.0), B: (0.0, 2.0), C: (3.0, 0.0), D: (0.0, -2.0)}, S)


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
    "kite": b_kite,
    "regular_polygon": b_regular_polygon,
}


def _need_name(name):
    if not name:
        raise ValueError("Thiếu tên điểm")
    return name


# ================================================================ v6: MỞ RỘNG
# Đánh dấu góc/cạnh, đường tròn nâng cao, hình không gian, mặt phẳng toạ độ.

AUTO_MARKS = True  # tự đánh dấu góc vuông / cạnh bằng nhau cho các hình đặc biệt
SPECIAL_FIGS = {"semicircle", "two_circles", "tetrahedron", "pyramid", "prism", "cube", "coordinate"}
NON_POLYGON_FIGS = {"circle", "none"} | SPECIAL_FIGS
TRI_FIGS = {"triangle", "right_triangle", "right_isosceles_triangle",
            "isosceles_triangle", "equilateral_triangle"}
C_OBL = (0.45, 0.35)  # hệ số phép chiếu xiên: trục sâu đi lên-phải


def parse_names(x):
    """Tên điểm: "ABC" -> A,B,C ; "A'B'" -> A',B' ; danh sách thì giữ nguyên."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(i).strip().replace("’", "'").replace("′", "'") for i in x if str(i).strip()]
    return re.findall(r"[A-Z]'*", str(x).replace("’", "'").replace("′", "'"))


def split_key(k):
    m = re.fullmatch(r"([A-Z]'?)([A-Z]'?)", str(k))
    return (m.group(1), m.group(2)) if m else None


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vunit(a):
    L = math.hypot(a[0], a[1])
    return (a[0] / L, a[1] / L) if L > 1e-12 else (1.0, 0.0)


def vang(a):
    return math.degrees(math.atan2(a[1], a[0]))


def _angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _size(pts):
    if len(pts) < 2:
        return 4.0
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    return max(max(xs) - min(xs), max(ys) - min(ys)) or 4.0


def _num(v, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if x > 0 else default


def _outward(pts, m, off):
    cx = sum(p[0] for p in pts.values()) / len(pts)
    cy = sum(p[1] for p in pts.values()) / len(pts)
    dx, dy = m[0] - cx, m[1] - cy
    d = math.hypot(dx, dy) or 1.0
    return (m[0] + dx / d * off, m[1] + dy / d * off)


def line_circle(p, v, c, r):
    """Tham số t của giao điểm đường thẳng p + t*v (v đơn vị) với đường tròn (c, r)."""
    w = vsub(p, c)
    b = w[0] * v[0] + w[1] * v[1]
    cc = w[0] ** 2 + w[1] ** 2 - r * r
    disc = b * b - cc
    if disc < 0:
        return []
    s = math.sqrt(disc)
    return [-b - s, -b + s]


def spread_angles(taken, k, start=100.0):
    """Chọn k góc (độ) cách xa các góc đã có và cách xa nhau."""
    out, ex = [], list(taken)
    for _ in range(k):
        best, bk = start, None
        for a in range(0, 360, 5):
            key = (min((_angdiff(a, e) for e in ex), default=360.0), -_angdiff(a, start))
            if bk is None or key > bk:
                best, bk = a, key
        out.append(best)
        ex.append(best)
    return out


def place_on_circle(pts, c, r, start=100.0):
    ex = [vang(vsub(p, c)) for p in pts.values() if abs(math.dist(p, c) - r) < 0.03 * r]
    a = spread_angles(ex, 1, start)[0]
    return (c[0] + r * math.cos(math.radians(a)), c[1] + r * math.sin(math.radians(a)))


class Ctx:
    """Trạng thái đang dựng hình, dùng chung cho các hàm mở rộng."""

    def __init__(self, pts, segs, shapes, texts, lines, drawn, counters, get_line, make_line):
        self.pts, self.segs, self.shapes, self.texts = pts, segs, shapes, texts
        self.lines, self.drawn, self.counters = lines, drawn, counters
        self.get_line, self.make_line = get_line, make_line
        self.thin, self.dashed, self.arrows, self.fills = [], [], [], []
        self.circles, self.labels, self.graphs = {}, {}, []
        self.axes = False

    def absorb(self, sp):
        self.dashed.extend(sp.get("dashed", []))
        self.circles.update(sp.get("circles", {}))
        self.labels.update(sp.get("labels", {}))
        self.graphs.extend(sp.get("graphs", []))
        self.axes = bool(sp.get("axes", False))


# ---------------------------------------------------------------- đánh dấu

def right_pts(ctx, V, P, Q):
    s = 0.07 * _size(ctx.pts)
    e1, e2 = vunit(vsub(P, V)), vunit(vsub(Q, V))
    a = (V[0] + e1[0] * s, V[1] + e1[1] * s)
    c = (V[0] + e2[0] * s, V[1] + e2[1] * s)
    ctx.thin.append([a, (a[0] + e2[0] * s, a[1] + e2[1] * s), c])


def tick_pts(ctx, P, Q, n=1):
    sz = _size(ctx.pts)
    n = int(n)
    d = vunit(vsub(Q, P))
    nv = nrm(d)
    m = mid(P, Q)
    for i in range(n):
        off = (i - (n - 1) / 2) * 0.04 * sz
        cx, cy = m[0] + d[0] * off, m[1] + d[1] * off
        s = 0.03 * sz
        ctx.thin.append([(cx - nv[0] * s, cy - nv[1] * s), (cx + nv[0] * s, cy + nv[1] * s)])


def angle_arc(ctx, V, P, Q, label=None, arcs=1):
    t1, t2 = vang(vsub(P, V)), vang(vsub(Q, V))
    sweep = (t2 - t1) % 360.0
    if sweep > 180.0:
        t1, sweep = t2, 360.0 - sweep
    sz = _size(ctx.pts)
    arcs = int(arcs)
    for k in range(arcs):
        ctx.shapes.append({"type": "arc", "cx": V[0], "cy": V[1], "r": (0.10 + 0.025 * k) * sz,
                           "t1": t1, "t2": t1 + sweep, "color": "tab:red", "lw": 1.3})
    if label:
        rr = (0.10 + 0.025 * (arcs - 1)) * sz * 1.9
        a = math.radians(t1 + sweep / 2)
        ctx.texts.append({"x": V[0] + rr * math.cos(a), "y": V[1] + rr * math.sin(a),
                          "text": str(label)})


def auto_marks(ctx, fig, n, data):
    pts = ctx.pts
    R = lambda v, p, q: right_pts(ctx, pts[v], pts[p], pts[q])
    T = lambda p, q, k=1: tick_pts(ctx, pts[p], pts[q], k)
    if AUTO_MARKS:
        if fig in ("right_triangle", "right_isosceles_triangle"):
            _, r, p, q = split_right(n, data)
            R(r, p, q)
            if fig == "right_isosceles_triangle":
                T(r, p)
                T(r, q)
        elif fig == "isosceles_triangle":
            A, B, C = n[:3]
            a = str(data.get("apex") or A)
            p, q = [x for x in (A, B, C) if x != a]
            T(a, p)
            T(a, q)
        elif fig == "equilateral_triangle":
            A, B, C = n[:3]
            T(A, B)
            T(B, C)
            T(C, A)
        elif fig in ("rectangle", "square"):
            A, B, C, D = n[:4]
            R(A, B, D)
            R(B, A, C)
            R(C, B, D)
            R(D, C, A)
            if fig == "square":
                T(A, B)
                T(B, C)
                T(C, D)
                T(D, A)
        elif fig == "rhombus":
            A, B, C, D = n[:4]
            T(A, B)
            T(B, C)
            T(C, D)
            T(D, A)
        elif fig == "right_trapezoid":
            A, B, C, D = n[:4]
            R(A, B, D)
            R(D, A, C)
        elif fig == "isosceles_trapezoid":
            A, B, C, D = n[:4]
            T(A, D)
            T(B, C)
        elif fig == "kite":
            A, B, C, D = n[:4]
            T(A, B)
            T(A, D)
            T(C, B, 2)
            T(C, D, 2)
    # các góc mà đề cho số đo
    if fig in TRI_FIGS:
        for x in n[:3]:
            v = data.get("angle_" + x)
            if v is not None and abs(float(v) - 90.0) > 1e-9:
                o = [y for y in n[:3] if y != x]
                angle_arc(ctx, pts[x], pts[o[0]], pts[o[1]], f"{float(v):g}°")
    elif fig in ("parallelogram", "rhombus") and data.get("angle") is not None:
        A, B, C, D = n[:4]
        angle_arc(ctx, pts[A], pts[B], pts[D], f"{float(data['angle']):g}°")


def foot_marks(ctx, frm, name, ln):
    """Ký hiệu vuông góc tại chân đường vuông góc; nối dài nét đứt nếu chân nằm ngoài đoạn."""
    H, F = ctx.pts[name], ctx.pts[frm]
    if math.dist(F, H) < 1e-9:
        return
    tgt = vsub(ln["p"], H)
    along = vunit(tgt) if math.hypot(*tgt) > 1e-6 else ln["v"]
    right_pts(ctx, H, F, (H[0] + along[0], H[1] + along[1]))
    if ln.get("kind") == "through":
        half = ln["scale"] / 2
        s = (H[0] - ln["p"][0]) * ln["v"][0] + (H[1] - ln["p"][1]) * ln["v"][1]
        if abs(s) > half + 1e-9:
            sg = 1.0 if s > 0 else -1.0
            E = (ln["p"][0] + ln["v"][0] * half * sg, ln["p"][1] + ln["v"][1] * half * sg)
            ctx.dashed.append([E, H])


def ortho_feet(ctx, e, of):
    """Chân ba đường cao (nếu đề đặt tên): feet = [D, E, F] ứng với đỉnh of[0], of[1], of[2]."""
    for i, nm in enumerate(parse_names(e.get("feet"))[:3]):
        v = of[i]
        ln = ctx.get_line([of[(i + 1) % 3], of[(i + 2) % 3]])
        P = ctx.pts[v]
        ctx.pts[nm] = foot(P, ln["p"], (ln["p"][0] + ln["v"][0], ln["p"][1] + ln["v"][1]))
        ctx.segs.append([P, ctx.pts[nm]])
        foot_marks(ctx, v, nm, ln)


def segment_param(ctx, e, p, q):
    """Vị trí điểm trên đoạn pq (0..1 tính từ p): theo ratio, dist, hoặc điều kiện về góc."""
    ang = e.get("angle")
    if isinstance(ang, dict):
        V, O = ctx.pts[str(ang["vertex"])], ctx.pts[str(ang["other"])]
        target = float(ang["value"])

        def f(t):
            X = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            return _angdiff(vang(vsub(O, V)), vang(vsub(X, V))) - target

        pt, pf = 0.0, f(0.0)
        for i in range(1, 401):
            t = i / 400.0
            cf = f(t)
            if pf == 0:
                return pt
            if pf * cf < 0:
                lo, hi = pt, t
                for _ in range(60):
                    m = (lo + hi) / 2
                    if f(lo) * f(m) <= 0:
                        hi = m
                    else:
                        lo = m
                return (lo + hi) / 2
            pt, pf = t, cf
        raise ValueError("Không có điểm trên đoạn thoả điều kiện góc")
    L = math.dist(p, q)
    if e.get("dist") and L > 1e-9:
        return float(e["dist"]) / L
    return float(e.get("ratio") or 0.4)


# ---------------------------------------------------------------- extras mở rộng

def graph_y(g, x):
    k = g["kind"]
    if k == "linear":
        return g["a"] * x + g["b"]
    if k == "quadratic":
        return g["a"] * x * x + g["b"] * x + g["c"]
    if k == "horizontal":
        return g["y"]
    return None


def ext_extra(ctx, t, name, e):
    """Các loại extras mở rộng. Trả về True nếu đã xử lý."""
    pts = ctx.pts
    raw = e.get("of")
    of = parse_names(raw) if isinstance(raw, (str, list)) else []

    if t == "bisector_foot":  # chân đường phân giác trong từ đỉnh "from" xuống cạnh "of"
        frm, p, q = str(e["from"]), of[0], of[1]
        ab, ac = math.dist(pts[frm], pts[p]), math.dist(pts[frm], pts[q])
        k = ab / (ab + ac)
        D = (pts[p][0] + (pts[q][0] - pts[p][0]) * k, pts[p][1] + (pts[q][1] - pts[p][1]) * k)
        pts[_need_name(name)] = D
        ctx.segs.append([pts[frm], D])
        angle_arc(ctx, pts[frm], pts[p], D)
        angle_arc(ctx, pts[frm], D, pts[q])
        return True
    if t == "reflect":  # điểm đối xứng qua một điểm hoặc qua một đường thẳng
        P, over = pts[str(e["point"])], e.get("over")
        if isinstance(over, str) and over in pts:
            O = pts[over]
            R = (2 * O[0] - P[0], 2 * O[1] - P[1])
        else:
            ln = ctx.get_line(over)
            H = foot(P, ln["p"], (ln["p"][0] + ln["v"][0], ln["p"][1] + ln["v"][1]))
            R = (2 * H[0] - P[0], 2 * H[1] - P[1])
        pts[_need_name(name)] = R
        return True
    if t == "parallelogram_vertex":  # ABCD là hình bình hành: D = A + C - B
        A, B, C = [pts[x] for x in of[:3]]
        pts[_need_name(name)] = (A[0] + C[0] - B[0], A[1] + C[1] - B[1])
        return True
    if t == "center_of":  # trung bình các điểm (tâm hình chữ nhật, hình bình hành...)
        P = [pts[x] for x in of]
        pts[_need_name(name)] = (sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P))
        return True
    if t == "extend":  # E nằm trên tia from->through, cách from = ratio * |from through|
        A, B = pts[str(e["from"])], pts[str(e["through"])]
        k = float(e.get("ratio") or 1.5)
        E = (A[0] + (B[0] - A[0]) * k, A[1] + (B[1] - A[1]) * k)
        pts[_need_name(name)] = E
        ctx.segs.append([B, E])
        return True
    if t == "incircle":  # tâm + đường tròn nội tiếp (+ tiếp điểm nếu đề đặt tên)
        A, B, C = [pts[x] for x in of[:3]]
        I = incenter(A, B, C)
        a, b, c = math.dist(B, C), math.dist(C, A), math.dist(A, B)
        r = abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) / (a + b + c)
        nm = name or "I"
        pts[nm] = I
        ctx.shapes.append({"type": "circle", "cx": I[0], "cy": I[1], "r": r,
                           "color": "tab:green", "fill": False})
        ctx.circles[nm] = (I, r)
        for tn, (P, Q) in zip(parse_names(e.get("touch"))[:3], ((B, C), (C, A), (A, B))):
            pts[tn] = foot(I, P, Q)
        return True
    if t == "circle":  # thêm một đường tròn: tâm + qua điểm | đường kính | bán kính số
        cn = str(e.get("center") or "")
        dia = parse_names(e.get("diameter"))
        if len(dia) >= 2:
            c = mid(pts[dia[0]], pts[dia[1]])
            r = math.dist(pts[dia[0]], pts[dia[1]]) / 2
            if cn and cn not in pts:
                pts[cn] = c
        else:
            c = pts[cn]
            thr = e.get("through")
            r = math.dist(c, pts[str(thr)]) if thr else _num(e.get("r"), 2.0)
        ctx.shapes.append({"type": "circle", "cx": c[0], "cy": c[1], "r": r,
                           "color": "tab:green", "fill": False})
        ctx.circles[cn or name or "c%d" % len(ctx.circles)] = (c, r)
        return True
    if t == "on_circle":  # điểm thuộc đường tròn tâm "of"
        c, r = ctx.circles[str(raw)]
        pts[_need_name(name)] = place_on_circle(pts, c, r)
        return True
    if t == "tangents":  # hai tiếp tuyến từ M tới đường tròn tâm "of", tiếp điểm = names
        M = str(e["from"])
        c, r = ctx.circles[str(raw)]
        if M not in pts:
            pts[M] = (c[0] + 2.0 * r, c[1])
        d = math.dist(pts[M], c)
        if d <= r * 1.001:
            raise ValueError("Điểm nằm trong đường tròn, không kẻ được tiếp tuyến")
        phi = math.atan2(pts[M][1] - c[1], pts[M][0] - c[0])
        al = math.acos(r / d)
        for nm, sg in zip(parse_names(e.get("names"))[:2], (1, -1)):
            T = (c[0] + r * math.cos(phi + sg * al), c[1] + r * math.sin(phi + sg * al))
            pts[nm] = T
            ctx.segs.append([pts[M], T])
            right_pts(ctx, T, c, pts[M])
        return True
    if t == "secant":  # cát tuyến từ M cắt đường tròn tại names (A gần M hơn B)
        M = str(e["from"])
        c, r = ctx.circles[str(raw)]
        if M not in pts:
            pts[M] = (c[0] + 2.2 * r, c[1])
        Mp = pts[M]
        v = vunit(vsub((c[0], c[1] + 0.35 * r), Mp))
        hits = sorted(line_circle(Mp, v, c, r))
        nms = parse_names(e.get("names"))
        if len(hits) < 2 or hits[0] <= 0 or len(nms) < 2:
            raise ValueError("Không dựng được cát tuyến")
        for nm, tt in zip(nms[:2], hits[:2]):
            pts[nm] = (Mp[0] + v[0] * tt, Mp[1] + v[1] * tt)
        ctx.segs.append([Mp, pts[nms[1]]])
        return True
    if t == "line_circle":  # giao của đường thẳng "line" với đường tròn tâm "of"
        ln = ctx.get_line(e["line"])
        c, r = ctx.circles[str(raw)]
        hits = sorted(line_circle(ln["p"], ln["v"], c, r))
        if not hits:
            raise ValueError("Đường thẳng không cắt đường tròn")
        for nm, tt in zip(parse_names(e.get("names")), hits):
            pts[nm] = (ln["p"][0] + ln["v"][0] * tt, ln["p"][1] + ln["v"][1] * tt)
        return True
    if t == "ray":  # tia from -> through
        A, B = pts[str(e["from"])], pts[str(e["through"])]
        ln = {"p": A, "v": vunit(vsub(B, A)), "scale": math.dist(A, B), "kind": "free", "ray": True}
        if name:
            ctx.lines[name] = ln
        ctx.drawn.append({**ln, "name": name})
        return True
    if t == "graph_intersection":  # giao điểm hai đồ thị (bậc nhất / bậc hai) trong mặt phẳng Oxy
        gs = [g for nm in e["of"] for g in ctx.graphs if g.get("name") == nm]
        if len(gs) < 2:
            raise ValueError("Thiếu đồ thị để tìm giao điểm")

        def coef(g):
            if g["kind"] == "linear":
                return (0.0, g["a"], g["b"])
            if g["kind"] == "quadratic":
                return (g["a"], g["b"], g["c"])
            raise ValueError("Chỉ hỗ trợ giao điểm của đồ thị bậc nhất/bậc hai")

        a1, b1, c1 = coef(gs[0])
        a2, b2, c2 = coef(gs[1])
        qa, qb, qc = a1 - a2, b1 - b2, c1 - c2
        if abs(qa) < 1e-12:
            if abs(qb) < 1e-12:
                raise ValueError("Hai đồ thị song song hoặc trùng nhau")
            xs = [-qc / qb]
        else:
            disc = qb * qb - 4 * qa * qc
            if disc < 0:
                raise ValueError("Hai đồ thị không cắt nhau")
            s = math.sqrt(disc)
            xs = sorted({(-qb - s) / (2 * qa), (-qb + s) / (2 * qa)})
        for nm, x in zip(parse_names(e.get("names")), xs):
            y = a1 * x * x + b1 * x + c1
            pts[nm] = (x, y)
            ctx.labels[nm] = f"{nm}({x:.3g};{y:.3g})"
        return True
    return False


def side_label_texts(pts, sides):
    out = []
    if not pts:
        return out
    size = _size(pts)
    for key, v in sides.items():
        nn = split_key(key)
        if not nn or nn[0] not in pts or nn[1] not in pts or nn[0] == nn[1]:
            continue
        x, y = _outward(pts, mid(pts[nn[0]], pts[nn[1]]), 0.07 * size)
        out.append({"x": x, "y": y, "text": f"{v:g}" if isinstance(v, (int, float)) else str(v)})
    return out


def post_top(ctx, data):
    """Các khoá cấp cao: dashed, vectors, fill, marks, side_labels."""
    pts = ctx.pts
    for s in data.get("dashed") or []:
        ch = [pts[x] for x in parse_names(s) if x in pts]
        if len(ch) >= 2:
            ctx.dashed.append(ch)
    for s in data.get("vectors") or []:
        ch = [pts[x] for x in parse_names(s) if x in pts]
        if len(ch) >= 2:
            ctx.arrows.append(ch[:2])
    for f in data.get("fill") or []:
        names = parse_names(f.get("points") if isinstance(f, dict) else f)
        color = (f.get("color") if isinstance(f, dict) else None) or "lightblue"
        if not is_color_like(color):
            color = "lightblue"
        ch = [pts[x] for x in names if x in pts]
        if len(ch) >= 3:
            ctx.fills.append({"pts": ch, "color": color})
    for m in data.get("marks") or []:
        t = m.get("type")
        if t == "right_angle":
            b = parse_names(m.get("between"))
            right_pts(ctx, pts[str(m["at"])], pts[b[0]], pts[b[1]])
        elif t == "equal_sides":
            for sd in m.get("sides") or []:
                nn = parse_names(sd)
                if len(nn) >= 2:
                    tick_pts(ctx, pts[nn[0]], pts[nn[1]], m.get("ticks") or 1)
        elif t == "angle":
            b = parse_names(m.get("between"))
            angle_arc(ctx, pts[str(m["at"])], pts[b[0]], pts[b[1]], m.get("label"), m.get("arcs") or 1)
        elif t == "equal_angles":
            for tri in m.get("angles") or []:
                nn = parse_names(tri)
                angle_arc(ctx, pts[nn[1]], pts[nn[0]], pts[nn[2]], None, m.get("arcs") or 1)
    for key, text in (data.get("side_labels") or {}).items():
        nn = parse_names(key)
        if len(nn) >= 2 and nn[0] in pts and nn[1] in pts:
            x, y = _outward(pts, mid(pts[nn[0]], pts[nn[1]]), 0.07 * _size(pts))
            ctx.texts.append({"x": x, "y": y, "text": str(text)})


# ---------------------------------------------------------------- hình đặc biệt

def sp_semicircle(names, data):
    o = str(data.get("center") or "O")
    A, B = (list(names) + ["A", "B"])[:2]
    r = _num(data.get("r"), 3.0)
    pts = {A: (-r, 0.0), B: (r, 0.0), o: (0.0, 0.0)}
    for nm, a in zip(parse_names(data.get("on_circle")), [60, 120, 90, 45, 135, 75, 105]):
        pts[nm] = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
    return {"pts": pts, "segs": [[(-r, 0.0), (r, 0.0)]],
            "shapes": [{"type": "arc", "cx": 0.0, "cy": 0.0, "r": r, "t1": 0.0, "t2": 180.0,
                        "color": "tab:blue", "lw": 2}],
            "circles": {o: ((0.0, 0.0), r)}}


def sp_two_circles(data):
    cn = parse_names(data.get("centers"))
    if len(cn) < 2:
        cn = ["O", "O'"]
    o1, o2 = cn[:2]
    r1, r2 = _num(data.get("r1"), 3.0), _num(data.get("r2"), 2.0)
    rel = str(data.get("relation") or "intersect")
    big, small = max(r1, r2), min(r1, r2)
    dd = {"intersect": big, "tangent_external": r1 + r2, "tangent_internal": big - small,
          "outside": 1.3 * (r1 + r2), "inside": 0.4 * (big - small)}.get(rel)
    if dd is None:
        raise ValueError("relation không hợp lệ: " + rel)
    if rel in ("tangent_internal", "inside") and big - small < 1e-9:
        raise ValueError("Hai đường tròn bằng nhau không thể lồng nhau")
    pts = {o1: (0.0, 0.0), o2: (dd, 0.0)}
    if rel == "intersect":
        a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
        h = root(r1 * r1 - a * a, "hai đường tròn không cắt nhau")
        common = [(a, h), (a, -h)]
    elif rel == "tangent_external":
        common = [(r1, 0.0)]
    elif rel == "tangent_internal":
        common = [(r1, 0.0)] if r1 >= r2 else [(-r1, 0.0)]
    else:
        common = []
    for nm, p in zip(parse_names(data.get("common")), common):
        pts[nm] = p
    return {"pts": pts, "segs": [],
            "shapes": [{"type": "circle", "cx": 0.0, "cy": 0.0, "r": r1, "color": "tab:blue", "fill": False},
                       {"type": "circle", "cx": dd, "cy": 0.0, "r": r2, "color": "tab:green", "fill": False}],
            "circles": {o1: ((0.0, 0.0), r1), o2: ((dd, 0.0), r2)}}


def _cross3(u, w):
    return (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])


def _dot3(u, w):
    return u[0] * w[0] + u[1] * w[1] + u[2] * w[2]


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _poly3d(verts, faces):
    """Chiếu xiên khối đa diện lồi; cạnh bị che (mọi mặt kề đều quay lưng) vẽ nét đứt."""
    c, s = C_OBL
    cen = tuple(sum(v[i] for v in verts.values()) / len(verts) for i in range(3))
    vis = []
    for f in faces:
        P = [verts[x] for x in f]
        N = _cross3(_sub3(P[1], P[0]), _sub3(P[2], P[0]))
        fc = tuple(sum(p[i] for p in P) / len(P) for i in range(3))
        if _dot3(N, _sub3(fc, cen)) < 0:
            N = (-N[0], -N[1], -N[2])
        vis.append(_dot3(N, (c, -1.0, s)) > 1e-9)
    edges = {}
    for f, v in zip(faces, vis):
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edges.setdefault(frozenset((a, b)), [(a, b), []])[1].append(v)
    pts2 = {k: (x + c * y, z + s * y) for k, (x, y, z) in verts.items()}
    solid, hidden = [], []
    for (a, b), vs in edges.values():
        (solid if any(vs) else hidden).append([pts2[a], pts2[b]])
    return pts2, solid, hidden


def sp_3d(fig, n, S, data):
    hk = data.get("height")
    if fig == "tetrahedron":
        A, B, C, D = n[:4]  # A là đỉnh, đáy BCD
        w = S(B, C) or 4.0
        h = float(hk or 3.8)
        V = {B: (0.0, 0.0, 0.0), C: (w, 0.0, 0.0), D: (0.8 * w, 3.0, 0.0), A: (0.6 * w, 1.0, h)}
        F = [[A, B, C], [A, C, D], [A, D, B], [B, C, D]]
    elif fig == "pyramid":
        S0, base = n[0], n[1:]
        nb = len(base)
        if nb not in (3, 4):
            raise ValueError("Đáy hình chóp phải có 3 hoặc 4 đỉnh")
        w = S(base[0], base[1]) or 4.0
        if nb == 4:
            d = w if str(data.get("base") or "") == "square" else (S(base[0], base[3]) or 3.0)
            V = {base[0]: (0.0, 0.0, 0.0), base[1]: (w, 0.0, 0.0),
                 base[2]: (w, d, 0.0), base[3]: (0.0, d, 0.0)}
        else:
            V = {base[0]: (0.0, 0.0, 0.0), base[1]: (w, 0.0, 0.0), base[2]: (0.8 * w, 3.0, 0.0)}
        over = str(data.get("apex_over") or "center")
        if over in base:
            ax, ay = V[over][0], V[over][1]
            h = float(hk or S(S0, over) or 4.0)
        else:
            ax = sum(V[x][0] for x in base) / nb
            ay = sum(V[x][1] for x in base) / nb
            h = float(hk or 4.0)
        V[S0] = (ax, ay, h)
        F = [list(base)] + [[S0, base[i], base[(i + 1) % nb]] for i in range(nb)]
    else:  # prism / cube
        nb = len(n) // 2
        if nb not in (3, 4) or len(n) != 2 * nb:
            raise ValueError("Cần 6 hoặc 8 tên đỉnh (đáy rồi đến mặt trên)")
        base, top = n[:nb], n[nb:]
        if fig == "cube":
            w = float(data.get("side") or S(base[0], base[1]) or 3.5)
            d = h = w
        else:
            w = S(base[0], base[1]) or 4.0
            d = (S(base[0], base[3]) if nb == 4 else None) or 3.0
            h = float(hk or S(base[0], top[0]) or 4.0)
        B0 = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)] if nb == 4 else \
             [(0.0, 0.0), (w, 0.0), (0.8 * w, d)]
        V = {}
        for nm, (x, y) in zip(base, B0):
            V[nm] = (x, y, 0.0)
        for nm, (x, y) in zip(top, B0):
            V[nm] = (x, y, h)
        F = [list(base), list(top)] + [[base[i], base[(i + 1) % nb], top[(i + 1) % nb], top[i]]
                                        for i in range(nb)]
    pts2, solid, hidden = _poly3d(V, F)
    return {"pts": pts2, "segs": solid, "dashed": hidden,
            "texts": side_label_texts(pts2, get_sides(data))}


def sp_coordinate(data):
    pts, labels = {}, {}
    show = data.get("show_coords", True)
    for k, v in (data.get("points") or {}).items():
        x, y = float(v[0]), float(v[1])
        pts[str(k)] = (x, y)
        if show:
            labels[str(k)] = f"{k}({x:g};{y:g})"
    graphs = []
    for g in data.get("graphs") or []:
        kind = str(g.get("kind") or "linear")
        if kind not in ("linear", "quadratic", "vertical", "horizontal"):
            raise ValueError("Loại đồ thị chưa hỗ trợ: " + kind)
        gv = lambda key, d: float(g[key]) if g.get(key) is not None else d
        graphs.append({"kind": kind, "a": gv("a", 1.0), "b": gv("b", 0.0), "c": gv("c", 0.0),
                       "x": gv("x", 0.0), "y": gv("y", 0.0), "name": str(g.get("name") or "")})
    return {"pts": pts, "segs": [], "labels": labels, "graphs": graphs, "axes": True}


def special_figure(fig, names, S, data):
    if fig == "semicircle":
        return sp_semicircle(names, data)
    if fig == "two_circles":
        return sp_two_circles(data)
    if fig == "coordinate":
        return sp_coordinate(data)
    return sp_3d(fig, names, S, data)


def build_figure(data: dict) -> dict:
    fig = data.get("figure")
    if fig not in ("circle", "none") and fig not in BUILDERS and fig not in SPECIAL_FIGS:
        raise ValueError("Loại hình chưa hỗ trợ: " + str(fig))

    names = parse_names(data.get("names"))
    sides = get_sides(data)
    S = lambda p, q: sides.get(p + q) or sides.get(q + p)
    shapes, segs, texts = [], [], []
    lines, drawn, counters = {}, [], {}  # lines: đường thẳng có tên; drawn: các đường sẽ vẽ
    sp = {}

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
            dia = parse_names(data.get("diameter"))
            if len(dia) >= 2:  # AB là đường kính: hai đầu đối xứng qua tâm
                for nm, a in zip(dia[:2], (150, 330)):
                    pts[nm] = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
                segs.append([pts[dia[0]], pts[dia[1]]])
                angs = spread_angles([150, 330], k)
            else:
                angs = {1: [90], 2: [150, 30], 3: [150, 30, 270]}.get(k) or \
                       [150 - i * 360 / k for i in range(k)]
            for nm, a in zip(on, angs):
                pts[nm] = (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        elif fig == "none":
            pts = {}
        elif fig in SPECIAL_FIGS:
            sp = special_figure(fig, names, S, data)
            pts = sp["pts"]
            segs.extend(sp.get("segs", []))
            shapes.extend(sp.get("shapes", []))
            texts.extend(sp.get("texts", []))
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

        ctx = Ctx(pts, segs, shapes, texts, lines, drawn, counters, get_line, make_line)
        ctx.absorb(sp)
        if fig == "circle":
            ctx.circles[o] = ((0.0, 0.0), r)
        if fig in BUILDERS:
            auto_marks(ctx, fig, names, data)

        for e in data.get("extras") or []:
            t, name = e.get("type"), str(e.get("name") or "")
            of = parse_names(e.get("of"))
            if t == "midpoint":
                pts[_need_name(name)] = mid(pts[of[0]], pts[of[1]])
            elif t == "foot":
                frm, ln = str(e["from"]), get_line(e["to"])
                pts[_need_name(name)] = foot(
                    pts[frm], ln["p"], (ln["p"][0] + ln["v"][0], ln["p"][1] + ln["v"][1]))
                segs.append([pts[frm], pts[name]])
                foot_marks(ctx, frm, name, ln)
            elif t == "on_segment":
                p, q = pts[of[0]], pts[of[1]]
                k = segment_param(ctx, e, p, q)
                pts[_need_name(name)] = (p[0] + k * (q[0] - p[0]), p[1] + k * (q[1] - p[1]))
            elif t == "centroid":
                P = [pts[x] for x in of[:3]]
                pts[_need_name(name)] = (sum(p[0] for p in P) / 3, sum(p[1] for p in P) / 3)
            elif t == "incenter":
                P = [pts[x] for x in of[:3]]
                pts[_need_name(name)] = incenter(*P)
            elif t == "orthocenter":
                P = [pts[x] for x in of[:3]]
                pts[_need_name(name)] = orthocenter(*P)
                ortho_feet(ctx, e, of)
            elif t == "circumcircle":
                name = name or "O"
                P = [pts[x] for x in of[:3]]
                o = circumcenter(*P)
                pts[name] = o
                shapes.append({"type": "circle", "cx": o[0], "cy": o[1],
                               "r": math.dist(o, P[0]), "color": "tab:green", "fill": False})
                ctx.circles[name] = (o, math.dist(o, P[0]))
            elif t == "line":
                th = e.get("through")
                tn = list(th) if isinstance(th, (str, list)) else []
                if e.get("perp_bisector_of"):
                    pq = parse_names(e["perp_bisector_of"])
                    P0, Q0 = pts[pq[0]], pts[pq[1]]
                    ln = {"p": mid(P0, Q0), "v": nrm(vunit(vsub(Q0, P0))),
                          "scale": math.dist(P0, Q0), "kind": "free"}
                elif e.get("tangent_at"):
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
                # Models sometimes encode "nối OM" as a named line instead of
                # putting OM in draw. A two-uppercase-letter name is a segment,
                # while names such as d/t remain infinite construction lines.
                segment_name = (len(name) == 2 and name.isupper()
                                and name[0] in pts and name[1] in pts)
                if segment_name:
                    segs.append([pts[name[0]], pts[name[1]]])
                else:
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
            elif ext_extra(ctx, t, name, e):
                pass
            else:
                raise ValueError("extras chưa hỗ trợ: " + str(t))

        for s in data.get("draw") or []:
            chain = [pts[x] for x in parse_names(s) if x in pts]
            if len(chain) >= 2:
                segs.append(chain)
        post_top(ctx, data)
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("Model trả thiếu/sai dữ liệu: " + repr(e))

    return {"points": pts, "segments": segs, "texts": texts, "shapes": shapes, "lines": drawn,
            "dashed": ctx.dashed, "thin": ctx.thin, "arrows": ctx.arrows, "fills": ctx.fills,
            "labels": ctx.labels, "axes": ctx.axes, "graphs": ctx.graphs}


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

def clip_line(p, v, x0, x1, y0, y1, ray=False):
    """Cắt đường thẳng (qua p, hướng v) theo hình chữ nhật khung nhìn."""
    lo, hi = (0.0 if ray else -1e9), 1e9
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
    seg = clip_line(L["p"], L["v"], *box, ray=bool(L.get("ray")))
    if not seg:
        return
    (xa, ya), (xb, yb) = seg
    if not L.get("ray") and (xb, yb) < (xa, ya):  # đầu mút "lớn hơn" ở bên phải/trên, để ghi tên ở đó
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


GRAPH_COLORS = ["tab:purple", "tab:orange", "tab:brown"]


def draw_graph(ax, g, box, color):
    x0, x1, y0, y1 = box
    if g["kind"] == "vertical":
        ax.plot([g["x"], g["x"]], [y0, y1], color=color, lw=1.8)
        if g.get("name"):
            ax.annotate(g["name"], (g["x"], y1 - 0.06 * (y1 - y0)), xytext=(8, 0),
                        textcoords="offset points", color=color, fontsize=13, fontstyle="italic")
        return
    xs = [x0 + (x1 - x0) * i / 400 for i in range(401)]
    ys = []
    for x in xs:
        y = graph_y(g, x)
        ys.append(y if (y is not None and y0 - 1e-9 <= y <= y1 + 1e-9) else float("nan"))
    ax.plot(xs, ys, color=color, lw=1.8)
    ok = [i for i, y in enumerate(ys) if y == y]
    if g.get("name") and ok:
        i = ok[int(len(ok) * 0.93)] if len(ok) > 20 else ok[-1]
        ax.annotate(g["name"], (xs[i], ys[i]), xytext=(8, 8), textcoords="offset points",
                    color=color, fontsize=13, fontstyle="italic")


def draw(d: dict, title: str, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 7))

    for f in d.get("fills", []):
        ax.add_patch(Polygon(f["pts"], closed=True, facecolor=f["color"],
                             edgecolor="none", alpha=0.35, zorder=0))

    for s in d["shapes"]:
        if s.get("type") == "arc":
            ax.add_patch(Arc((s["cx"], s["cy"]), 2 * s["r"], 2 * s["r"], theta1=s["t1"],
                             theta2=s["t2"], edgecolor=s["color"], lw=s.get("lw", 2)))
        else:
            ax.add_patch(Circle((s["cx"], s["cy"]), s["r"], fill=s.get("fill", False),
                                facecolor=s["color"] if s.get("fill") else "none",
                                edgecolor=s["color"], lw=s.get("lw", 2)))

    for chain in d["segments"]:
        xs, ys = zip(*chain)
        ax.plot(xs, ys, color="black", lw=2)
    for chain in d.get("dashed", []):
        xs, ys = zip(*chain)
        ax.plot(xs, ys, color="black", lw=1.6, linestyle="--")
    for chain in d.get("thin", []):
        xs, ys = zip(*chain)
        ax.plot(xs, ys, color="tab:red", lw=1.2)
    for chain in d.get("arrows", []):
        (xa, ya), (xb, yb) = chain[0], chain[-1]
        ax.plot([xa, xb], [ya, yb], alpha=0)
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    labels = d.get("labels", {})
    if d["points"]:
        cx = sum(p[0] for p in d["points"].values()) / len(d["points"])
        cy = sum(p[1] for p in d["points"].values()) / len(d["points"])
        for name, (x, y) in d["points"].items():
            ax.plot(x, y, "o", color="black", ms=5)
            dx, dy = x - cx, y - cy
            n = math.hypot(dx, dy)
            if n < 1e-9:
                dx, dy, n = -1.0, -1.0, math.sqrt(2)
            ax.annotate(labels.get(name, name), (x, y), xytext=(dx / n * 14, dy / n * 14),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=14 if name not in labels else 11, fontweight="bold", zorder=10,
                        bbox=dict(facecolor="white", edgecolor="none", pad=0.7))

    for t in d["texts"]:
        ax.plot(t["x"], t["y"], alpha=0)  # để autoscale tính cả vị trí text
        ax.text(t["x"], t["y"], t["text"], ha="center", va="center",
                fontsize=12, color="tab:red")

    # chốt khung nhìn trước (theo điểm/hình/đoạn đại diện của đường thẳng/đồ thị), rồi mới
    # vẽ đường thẳng, tia, đồ thị kéo dài tới mép khung
    for L in d.get("lines", []):
        hh = 0.5 * L.get("scale", 4.0)
        a0 = 0.0 if L.get("ray") else -hh
        ax.plot([L["p"][0] + L["v"][0] * a0, L["p"][0] + L["v"][0] * hh],
                [L["p"][1] + L["v"][1] * a0, L["p"][1] + L["v"][1] * hh], alpha=0)
    gx = d.get("graphs") or []
    if d.get("axes"):
        ax.plot([0], [0], alpha=0)
        xp = [p[0] for p in d["points"].values()]
        gx0, gx1 = min([-3.0] + [x - 1 for x in xp]), max([3.0] + [x + 1 for x in xp])
        for g in gx:
            if g["kind"] == "vertical":
                ax.plot([g["x"]], [0], alpha=0)
                continue
            for i in range(61):
                x = gx0 + (gx1 - gx0) * i / 60
                y = graph_y(g, x)
                if y is not None and abs(y) <= 10:
                    ax.plot([x], [y], alpha=0)
    if (d["points"] or d["shapes"] or d["segments"] or d["texts"] or d.get("lines")
            or d.get("axes") or d.get("dashed")):
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
    for k, g in enumerate(gx):
        draw_graph(ax, g, (x0, x1, y0, y1), GRAPH_COLORS[k % len(GRAPH_COLORS)])
    if d.get("axes"):  # trục toạ độ đi qua gốc, có mũi tên
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.plot(x1, 0, ">k", clip_on=False, ms=6)
        ax.plot(0, y1, "^k", clip_on=False, ms=6)
        ax.annotate("x", (x1, 0), xytext=(-2, 8), textcoords="offset points", fontsize=12)
        ax.annotate("y", (0, y1), xytext=(8, -4), textcoords="offset points", fontsize=12)
        if "O" not in d["points"]:
            ax.annotate("O", (0, 0), xytext=(-9, -12), textcoords="offset points", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=10)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  Đã lưu ảnh: {output_path}")
    plt.show()


def run(prompt: str, output_path: Path | None = None):
    normalized_prompt = normalize_prompt(prompt)
    last = None
    for i in range(1, MAX_RETRY + 1):
        temp = 0 if i == 1 else 0.5  # temp=0 thì retry ra y hệt, nên các lần sau tăng nhẹ
        print(f"[{i}/{MAX_RETRY}] đang hỏi {MODEL}...")
        try:
            raw = ask_model(normalized_prompt, temp)
            print("  model trả:", raw.strip()[:500])
            data = enforce_explicit_figure(extract_json(raw), normalized_prompt)
            data = enforce_diagonal_construction(data, normalized_prompt)
            data = enforce_explicit_relations(data, normalized_prompt)
            data = remove_invented_measurements(data, normalized_prompt)
            draw(parse(data), prompt, output_path)
            return
        except Exception as e:
            last = e
            print("  lỗi:", repr(e))
    print("Thất bại:", last)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tạo hình học từ đề bài bằng Ollama.")
    parser.add_argument("prompt", nargs="*", help="Đề bài hình học")
    parser.add_argument("--out", type=Path, help="Đường dẫn PNG cần xuất")
    args = parser.parse_args()
    if args.prompt:
        target = args.out
        if target is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = OUTPUT_DIR / f"diagram-{stamp}.png"
        run(" ".join(args.prompt), target)
    else:
        while True:
            p = input("Đề (q để thoát): ").strip()
            if p.lower() in ("q", "quit", ""):
                break
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run(p, OUTPUT_DIR / f"diagram-{stamp}.png")