# Insert a "view English original" link at the top of every built page.
# Maps built-HTML path -> docs page -> original repo path on master.
import os, re, sys

SITE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mleng-site-zh"
REPO = "https://github.com/CodingPracticeGroup/ml-engineering/blob/master/"
BLOCK = ('<p class="en-origin-link">📄 <a href="{}" target="_blank" rel="noopener">'
         '本页英文原文（GitHub master）</a></p>')

def src_of(p):
    """Built html path -> original English source path in the repo."""
    if p == "index.html":
        return "README.md"
    if p.endswith("/index.html"):
        return p[: -len("/index.html")] + "/README.md"
    if p.endswith(".html"):
        return p[: -len(".html")] + ".md"
    return None

n = 0
for dirpath, _, files in os.walk(SITE):
    for fn in files:
        if not fn.endswith(".html") or fn == "404.html":
            continue
        full = os.path.join(dirpath, fn)
        rel = os.path.relpath(full, SITE).replace(os.sep, "/")
        src = src_of(rel)
        if not src:
            continue
        html = open(full, encoding="utf-8").read()
        if "en-origin-link" in html:
            continue
        block = BLOCK.format(REPO + src)
        # put right before the first heading of the article content
        m = re.search(r'<h1', html)
        if m:
            html = html[:m.start()] + block + html[m.start():]
        else:
            m = re.search(r'(<div class="md-content__inner[^"]*">)', html)
            if not m:
                print("SKIP (no anchor):", rel)
                continue
            html = html[:m.end()] + block + html[m.end():]
        open(full, "w", encoding="utf-8").write(html)
        n += 1
print(f"injected English-original link into {n} pages")
