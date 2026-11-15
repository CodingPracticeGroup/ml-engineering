# Site-wide link check on BUILT html: every local href/src must resolve to an
# existing file, and every #fragment must exist as an id in the target page.
import glob, os, re, sys

SITE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mleng-site-zh"
BASE = "/ml-engineering"  # site_url path prefix on GitHub Pages
ids_cache = {}

def ids_of(f):
    if f not in ids_cache:
        try:
            ids_cache[f] = set(re.findall(r'id="([^"]+)"', open(f, encoding="utf-8").read()))
        except OSError:
            ids_cache[f] = set()
    return ids_cache[f]

bad = []
checked = 0
for f in glob.glob(os.path.join(SITE, "**/*.html"), recursive=True):
    html = open(f, encoding="utf-8").read()
    for m in re.finditer(r'(?:href|src)="([^"#]*)(#[^"]*)?"', html):
        path, frag = m.group(1), (m.group(2) or "")[1:]
        if path.startswith(("http:", "https:", "mailto:", "javascript:")):
            continue
        checked += 1
        tgt = os.path.join(SITE, path.lstrip("/")) if path.startswith("/") \
              else os.path.normpath(os.path.join(os.path.dirname(f), path or "."))
        if path.startswith(BASE + "/"):
            tgt = os.path.join(SITE, path[len(BASE) + 1:])
        elif path == BASE:
            tgt = os.path.join(SITE, "index.html")
        if path == "":
            tgt = f
        if not os.path.exists(tgt):
            bad.append((os.path.relpath(f, SITE), path, frag, "MISSING FILE")); continue
        if frag:
            if os.path.isdir(tgt):
                tgt = os.path.join(tgt, "index.html")
                if not os.path.exists(tgt):
                    bad.append((os.path.relpath(f, SITE), path, frag, "MISSING FILE")); continue
            if frag not in ids_of(tgt):
                bad.append((os.path.relpath(f, SITE), path, frag, "NO ANCHOR"))

print(f"local href/src checked: {checked}, problems: {len(bad)}")
for b in bad[:60]:
    print("  ", b[0], "->", b[1] + b[2], "|", b[3])
