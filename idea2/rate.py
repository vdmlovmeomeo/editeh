import unicodedata

def norm(s) -> str:
    s = str(s).casefold().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

# (title, từ khoá content, deadline, created_at)
EXPECTED = [
    ("esl", "language", "2026-11-20", "2026-11-16"),
    ("esl", "flyer", "2026-11-18", "2026-11-16"),
    ("esl", "workbook", "2026-11-18", "2026-11-16"),
    ("su", "tom tat", "2026-11-24", "2026-11-17"),
    ("toan", "ly thuyet", "2026-11-23", "2026-11-17"),
    ("toan", "hdot", "2026-11-23", "2026-11-17"),
    ("toan", "kscl", "2026-11-23", "2026-11-17"),
    ("van", "hdot", "2026-11-26", "2026-11-18"),
]

def score(tasks):
    hits = 0
    for title_k, content_k, dl, cr in EXPECTED:
        for t in tasks:
            if (
                title_k in norm(t.get("title", ""))
                and content_k in norm(t.get("content", ""))
                and t.get("deadline") == dl
                and t.get("created_at") == cr
            ):
                hits += 1
                break
    return hits, 0