# Normalize continuation-block indentation inside ordered lists.
#
# The English corpus indents continuation paragraphs under "1. " items by 3
# spaces (the marker's content column). CommonMark (GitHub) renders that inside
# the item; Python-Markdown (mkdocs) requires >=4 spaces, splits the item out,
# and restarts numbering in a new <ol> — showing two "1.". Fix: bump such
# 3-space continuations to 4 spaces in docs/*.md. Safe to re-run (idempotent).
import os, re, sys

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
OL_ITEM = re.compile(r"^( *)(\d+)([.)])( +)\S")
FENCE = re.compile(r"^\s*(```|~~~)")

def fix_text(text):
    lines = text.splitlines(keepends=True)
    out, fence, col = [], None, None  # col = required continuation indent
    for ln in lines:
        m = FENCE.match(ln)
        if m:
            if fence is None: fence = m.group(1)
            elif ln.lstrip().startswith(fence): fence = None
            out.append(ln); continue
        if fence is not None:
            out.append(ln); continue
        mo = OL_ITEM.match(ln)
        if mo:
            # top-level items set the required content column
            col = len(mo.group(1)) + len(mo.group(2)) + 1 + len(mo.group(4)) if not mo.group(1) else None
            out.append(ln); continue
        if ln.strip() == "":
            out.append(ln); continue
        if (col and ln.startswith(" " * col) and not ln.startswith(" " * (col + 1))
                and not re.match(r"^ *[" + r">*\-+#~`|[" + r"\d]", ln)):
            out.append(" " + ln)   # 3 -> 4 spaces
            continue
        if not ln.startswith(" "):
            col = None             # unindented paragraph ends the item context
        out.append(ln)
    return "".join(out)

changed = 0
for dirpath, _, files in os.walk(DOCS):
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(dirpath, fn)
        src = open(p, encoding="utf-8").read()
        dst = fix_text(src)
        if dst != src:
            open(p, "w", encoding="utf-8").write(dst)
            changed += 1
print(f"normalized continuation indent in {changed} md files")
