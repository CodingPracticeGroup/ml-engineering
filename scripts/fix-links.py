# Post-process links inside translated docs:
#  1) links resolving to root README.md -> index.md (root README became the home page)
#  2) links to content excluded from the site -> full URLs on the GitHub repo
import os, re, posixpath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
EXCLUDED = re.compile(r"^(?:todo\.md|SKILL\.md|stabs/|skills/|build/SESSION\.md|LICENSE-CC-BY-SA)")
GITHUB = "https://github.com/CodingPracticeGroup/ml-engineering/blob/master/"
LINK = re.compile(r"\]\(([^)\s]+?)(#[^)\s]*)?\)")

def to_upstream(rel):
    return GITHUB + rel

def rewrite_file(path):
    rel_doc = os.path.relpath(path, DOCS).replace(os.sep, "/")
    d = posixpath.dirname(rel_doc)
    src_text = open(path, encoding="utf-8").read()
    changes = []

    def sub(m):
        target, frag = m.group(1), m.group(2) or ""
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(("#", "mailto:")):
            return m.group(0)
        norm = posixpath.normpath(posixpath.join(d, target))
        if norm == "README.md":  # root README -> home page
            new = target[: -len("README.md")] + "index.md"
            changes.append((target, new))
            return "](" + new + frag + ")"
        # directory links like `(compute)` -> point at that dir's README.md page
        if norm not in ("", ".") and os.path.isdir(os.path.join(DOCS, norm)):
            if os.path.exists(os.path.join(DOCS, norm, "README.md")):
                new = target.rstrip("/") + "/README.md"
                changes.append((target, new))
                return "](" + new + frag + ")"
            return m.group(0)
        if not EXCLUDED.search(norm):
            return m.group(0)
        new = to_upstream(norm)
        changes.append((target, new))
        return "]" + "(" + new + frag + ")"

    out = LINK.sub(sub, src_text)
    if out != src_text:
        open(path, "w", encoding="utf-8").write(out)
    return rel_doc, changes

total = 0
for dirpath, dirnames, files in os.walk(DOCS):
    for fn in files:
        if fn.endswith(".md"):
            rel, ch = rewrite_file(os.path.join(dirpath, fn))
            for a, b in ch:
                print(f"  {rel}: {a} -> {b}")
                total += 1
print(f"rewrote {total} links")
