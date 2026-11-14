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
    prev_kind = None
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
            prev_kind = None
            out.append(ln); prev_nonblank = body; continue
        if fence is not None:
            out.append(ln); prev_nonblank = body; continue
        m = ITEM.match(body)
        if m and body.lstrip()[0] not in ">":
            indent = len(m.group(1))
            # Determine the real nesting parent by indentation, BEFORE any rule.
            # Pop parents that are not less-indented than this item.
            while stack and stack[-1][0] >= indent:
                stack.pop()
            # A list marker indented 1-3 spaces with NO enclosing list item is
            # top-level with stray leading whitespace (CommonMark reads it as a
            # top-level list; Python-Markdown would swallow it into the prior
            # paragraph as lazy continuation). Normalize to column 0.
            if not stack and 0 < indent < 4:
                body = body.lstrip()
                indent = 0
            # R1: new top-level list right after a paragraph -> blank line first
            if (indent == 0 and BARE_LIST_START.match(body.lstrip())
                    and is_para(prev_nonblank) and out and out[-1].strip() != ""):
                out.append("\n")
            # R3: an adjacent list of the other type (ol<->ul) makes
            # Python-Markdown merge it into the previous list's items.
            # A blank line + HTML comment splits them; invisible in the browser.
            # Fires whenever the top-level list type flips with no hard block
            # (heading/code/html) in between — a stray separator between two
            # genuinely separate lists is harmless.
            cur_kind = "ol" if body.lstrip()[0].isdigit() else "ul"
            if indent == 0:
                if prev_kind and prev_kind != cur_kind:
                    if out and out[-1].strip() != "":
                        out.append("\n")
                    out.append("<!---->\n\n")
                prev_kind = cur_kind
            # R2: indent to satisfy parent_indent + 4 when nesting
            want = stack[-1][0] + 4 if stack else 0
            if indent and want and indent < want:
                body = " " * want + body.lstrip()
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
            # a plain paragraph does NOT reset prev_kind: Python-Markdown
            # absorbs a following list into the previous one across it (that's
            # the bug R3 guards); only real block boundaries end the sequence.
            if HARD.match(body):
                prev_kind = None
        out.append(ln); prev_nonblank = body
    return "".join(out)


BOL = re.compile(r"^( *)(\d+)([.)]) ")
HARD = re.compile(r"^\s*(#{1,6}\s|<(?:!--|/?[a-zA-Z])|\||-{3,}|=)")


def merge_split_ordered(text):
    """R4: the author splits numbered steps with top-level code blocks or
    paragraphs ("1. step", fence, "2. step"). GitHub keeps numbering via
    start=N; Python-Markdown restarts at 1. Fold such in-between content into
    the preceding item (4-space indent) so the list continues in BOTH
    renderers. Only applied when the next marker's number continues the
    sequence, so genuine list endings are never touched."""
    lines = text.splitlines(keepends=True)
    out, fence, gap, last_num = [], None, [], None
    def flush(indent_gap=False):
        for g in gap:
            if indent_gap and g.strip() and (len(g) - len(g.lstrip())) < 4:
                out.append("    " + g)
            else:
                out.append(g)
        gap.clear()
    for ln in lines:
        body = ln.rstrip("\n")
        m = FENCE.match(body)
        if m:
            if fence is None:
                fence = m.group(1)
            elif body.lstrip().startswith(fence):
                fence = None
            (gap if last_num else out).append(ln)
            continue
        if fence is not None:
            (gap if last_num else out).append(ln)
            continue
        mo = BOL.match(body)
        if mo and not body.startswith(" "):
            n = int(mo.group(2))
            if last_num is not None and n == last_num + 1:
                flush(indent_gap=True)          # reconnect: content joins the item
                out.append(ln); last_num = n
            else:
                flush(); out.append(ln); last_num = n
            continue
        # bullets (any indent) inside a running numbered sequence are author
        # style for extra text between steps -> treat as gap, folded at +4
        if last_num is not None and re.match(r"^\s*[-*+] ", body):
            gap.append(ln)
            continue
        if body.strip() == "":
            (gap if last_num is not None else out).append(ln)
            continue
        if last_num is not None and not HARD.match(body):
            gap.append(ln)
            continue
        flush()
        last_num = None
        out.append(ln)
    flush()
    return "".join(out)

changed = 0
for dirpath, _, files in os.walk(DOCS):
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(dirpath, fn)
        src = open(p, encoding="utf-8").read()
        dst = merge_split_ordered(fix_text(src))
        if dst != src:
            open(p, "w", encoding="utf-8").write(dst)
            changed += 1
print(f"normalized list formatting in {changed} md files")
