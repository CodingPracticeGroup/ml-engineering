# Normalize list quirks in docs/*.md so Python-Markdown (mkdocs) renders the
# same lists GitHub/CommonMark does. Idempotent; run after merge-parts.py and
# before building the site.
#
#  R1  a list that directly follows a paragraph line gets a blank line inserted
#      (Python-Markdown never starts a list without a preceding blank line).
#  R2  nested list items and continuation paragraphs under a list item are
#      re-indented to parent_indent + 4 (Python-Markdown's fixed 4-space rule;
#      CommonMark instead aligns with the marker's content column, which the
#      English corpus uses: 2 or 3 spaces).
import os, re

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
FENCE = re.compile(r"^\s*(```|~~~)")
ITEM = re.compile(r"^( *)([-*+]|\d+[.)])( +)(\S.*)$")
BARE_LIST_START = re.compile(r"^(?:[-*+]|1[.)]) \S")

def is_para(text_line):
    s = text_line.strip()
    if not s:
        return False
    if s[0] in "#>|`!" or s.startswith(("<!--", "![", "<a", "</a", "=== ", "[{")):
        return False
    return not ITEM.match(text_line)

def fix_text(text):
    lines = text.splitlines(keepends=True)
    out, fence = [], None
    stack = []  # (indent, is_list_line) parent chain of list items
    prev_nonblank = ""
    for ln in lines:
        body = ln.rstrip("\n")
        if FENCE.match(body):
            if fence is None:
                fence = FENCE.match(body).group(1)
            elif body.lstrip().startswith(fence):
                fence = None
            stack = []
            out.append(ln); prev_nonblank = body; continue
        if fence is not None:
            out.append(ln); prev_nonblank = body; continue
        m = ITEM.match(body)
        if m and body.lstrip()[0] not in ">":
            indent = len(m.group(1))
            # R1: new top-level list right after a paragraph -> blank line first
            if (indent == 0 and BARE_LIST_START.match(body.lstrip())
                    and is_para(prev_nonblank) and out and out[-1].strip() != ""):
                out.append("\n")
            # R2: indent to satisfy parent_indent + 4 when nesting
            while stack and stack[-1][0] >= indent:
                stack.pop()
            want = stack[-1][0] + 4 if stack else 0
            if indent and indent < want and indent > stack[-1][0] if stack else False:
                body = " " * want + body[indent:]
                indent = want
            stack.append((indent, True))
            out.append(body + "\n"); prev_nonblank = body; continue
        if body.strip() == "":
            out.append(ln); continue
        # R2 continuation paragraph under a list item
        if stack:
            indent = len(body) - len(body.lstrip())
            want = stack[-1][0] + 4
            if 0 < indent < want and not re.match(r"^ *[>|]", body):
                body = " " * want + body.lstrip()
                out.append(body + "\n"); prev_nonblank = body; continue
        if not body.startswith(" "):
            stack = []
        out.append(ln); prev_nonblank = body
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
print(f"normalized list formatting in {changed} md files")
