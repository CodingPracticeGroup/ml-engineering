# Verify merged translations against sources: headings & code fences preserved,
# Chinese present, and how much untranslated English prose remains outside code.
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_order as co

ROOT = co.ROOT
DOCS = os.path.join(ROOT, "docs")
CJK = re.compile(r"[\u4e00-\u9fff]")
HEAD = re.compile(r"^#{1,6}\s")
FENCE = re.compile(r"^\s*(```|~~~)")

def code_fence_count(lines):
    fence, n = None, 0
    for ln in lines:
        m = FENCE.match(ln)
        if not m:
            continue
        if fence is None:
            fence = m.group(1); n += 1
        elif ln.lstrip().startswith(fence):
            fence = None
    return n

def strip_code(text):
    lines, out, fence = text.splitlines(), [], None
    for ln in lines:
        m = FENCE.match(ln)
        if fence is not None:
            if m and ln.lstrip().startswith(fence):
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        out.append(re.sub(r"`[^`]*`", " ", ln))
    return "\n".join(out)

def latin_words(text):
    return len(re.findall(r"[A-Za-z]{3,}", text))

report = []
bad = 0
for u in json.load(open(os.path.join(ROOT, "zh_work", "manifest.json"))):
    src = open(os.path.join(ROOT, u["src"]), encoding="utf-8").read()
    dstp = os.path.join(DOCS, u["dst"])
    if not os.path.exists(dstp):
        report.append(f"FAIL missing {u['dst']}"); bad += 1; continue
    dst = open(dstp, encoding="utf-8").read()
    s_lines, d_lines = src.splitlines(), dst.splitlines()
    s_h = sum(1 for l in s_lines if HEAD.match(l) and "```" not in l)
    d_h = sum(1 for l in d_lines if HEAD.match(l))
    s_f, d_f = code_fence_count(s_lines), code_fence_count(d_lines)
    cjk = len(CJK.findall(dst))
    src_prose = latin_words(strip_code(src))
    rem = latin_words(strip_code(dst))
    issues = []
    if cjk < 50:
        issues.append("no-Chinese")
    if d_h < s_h * 0.95:
        issues.append(f"headings {d_h}<{s_h}")
    if d_f != s_f:
        issues.append(f"fences {d_f}!={s_f}")
    if rem > 0.5 * max(src_prose, 1):
        issues.append(f"latin-remaining {rem}/{src_prose}")
    if issues:
        bad += 1
        report.append(f"REVIEW {u['dst']}: " + "; ".join(issues))
print(f"checked {len(json.load(open(os.path.join(ROOT,'zh_work','manifest.json'))))} files, flagged {bad}")
print("\n".join(report))
sys.exit(1 if any(r.startswith("FAIL") for r in report) else 0)
