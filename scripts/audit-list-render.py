# Whole-site list-rendering audit.
#
# For every translated page, render the markdown twice — once with a
# CommonMark renderer (GitHub semantics) and once with the Python-Markdown
# stack the site actually builds with — extract the nested list skeleton
# (ol/ul nesting, li counts) from each, and diff. Any difference is a place
# where our site will show a broken split/restarted list.
import os, re, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_order as co
from markdown_it import MarkdownIt
import markdown as pymd

DOCS = os.path.join(co.ROOT, "docs")
SITE = os.path.join(co.ROOT, "site")
CM = MarkdownIt("gfm-like").disable("linkify")
PYMD_EXT = ["toc", "attr_list", "admonition", "footnotes", "md_in_html",
            "tables", "sane_lists",
            "pymdownx.highlight", "pymdownx.inlinehilite", "pymdownx.superfences",
            "pymdownx.details", "pymdownx.tabbed"]

LIST = re.compile(r"<(/?)(ol|ul)([^>]*)>|<li[^>]*>|<pre>|</pre>|<code>|</code>")

def skeleton(html):
    """Visible numbering signature: for each <li> in document order, its depth
    and the label a reader would see ("3." / "-"). Structural differences that
    render identically (e.g. start= attrs vs continued lists) compare equal."""
    from html.parser import HTMLParser
    class Sig(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack, self.sig, self.skip = [], [], 0
        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if tag in ("pre", "code"):
                self.skip += 1; return
            if self.skip or tag not in ("ol", "ul", "li"):
                return
            if tag in ("ol", "ul"):
                start = int(a.get("start", 1))
                self.stack.append([tag, start])
                return
            # li: label from innermost open list; don't double-count nested li
            if self.stack:
                top = self.stack[-1]
                label = f"{top[1]}." if top[0] == "ol" else "-"
                if top[0] == "ol":
                    top[1] += 1
                self.sig.append((len(self.stack), label))
        def handle_endtag(self, tag):
            if tag in ("pre", "code"):
                self.skip = max(0, self.skip - 1); return
            if not self.skip and tag in ("ol", "ul") and self.stack:
                self.stack.pop()
    s = Sig(); s.feed(html)
    return s.sig

def clean_src(src):
    # drop custom heading anchors so both renderers see identical anchors
    return re.sub(r" ?\{#[^}]+\}", "", src)

ART = re.compile(r'<article[^>]*md-content__inner[^>]*>(.*?)</article>', re.S)

def body_of(relmd):
    """site/ built HTML for a docs page (use_directory_urls: false)."""
    if relmd.endswith("README.md"):
        h = os.path.join(SITE, os.path.dirname(relmd), "index.html")
    else:
        h = os.path.join(SITE, os.path.splitext(relmd)[0] + ".html")
    if not os.path.exists(h):
        return None
    m = ART.search(open(h, encoding="utf-8").read())
    if not m:
        return None
    html = m.group(1)
    # footnotes are standard-rendered and out of scope; drop from end
    k = re.search(r'<(?:div|section)[^>]*class="footnotes?"', html)
    return html[:k.start()] if k else html

def cm_body(src):
    html = CM.render(src)
    k = html.find('<section class="footnotes"')
    if k < 0:
        k = html.find('<ol class="footnotes"')
    return html[:k] if k >= 0 else html

bad = []
checked = 0
missing = []
for dirpath, _, files in os.walk(DOCS):
    for fn in sorted(files):
        if not fn.endswith(".md"):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, DOCS)
        built = body_of(rel)
        if built is None:
            missing.append(rel); continue
        src = clean_src(open(p, encoding="utf-8").read())
        try:
            cm = skeleton(cm_body(src))
            pm = skeleton(built)
        except Exception as e:
            bad.append((rel, "render error: " + str(e)[:80])); continue
        checked += 1
        if cm != pm:
            k = next(i for i in range(max(len(cm), len(pm)))
                     if (cm[i] if i < len(cm) else None) != (pm[i] if i < len(pm) else None))
            bad.append((rel, f"first diff at token {k}: commonmark={cm[max(0,k-1):k+3]} site={pm[max(0,k-1):k+3]}"))

print(f"pages compared: {checked}, list-skeleton mismatches: {len(bad)}")
if missing:
    print("no built page (skip):", ", ".join(missing[:5]), ("..." if len(missing) > 5 else ""))
for rel, why in bad:
    print(" -", rel)
    print("    ", why[:220])
