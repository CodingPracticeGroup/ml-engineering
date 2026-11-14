# Split oversized md files into contiguous line-range parts (~<=MAXW words each)
# so one translation agent handles at most one part, and emit zh_work/manifest.json.
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_order as co

MAXW = 4500          # soft cap on words per part
HARD_CAP = 9000      # split further only past this
HEAD = re.compile(r"^#{1,6}\s")
FENCE = re.compile(r"^\s*(```|~~~)")

ROOT = co.ROOT
WORK = os.path.join(ROOT, "zh_work")


def words(s):
    return len(s.split())


def heading_indices(lines):
    """Line indices that start a real (outside-code-fence) heading, plus 0."""
    idx, fence = [0], None
    for i, ln in enumerate(lines):
        m = FENCE.match(ln)
        if m:
            if fence is None:
                fence = m.group(1)
            elif ln.lstrip().startswith(fence):
                fence = None
            continue
        if fence is None and HEAD.match(ln) and i > 0:
            idx.append(i)
    return idx


def blocks(lines):
    idxs = heading_indices(lines)
    return [(a, b) for a, b in zip(idxs, idxs[1:] + [len(lines)])]


def split_big(lines, a, b):
    """A single block larger than HARD_CAP: split at ### headings, then by line count."""
    segs = [a] + [i for i in range(a, b)
                  if re.match(r"^###\s", lines[i])] + [b]
    parts = [(x, y) for x, y in zip(segs, segs[1:]) if y > x]
    out = []
    for x, y in parts:
        seg = lines[x:y]
        if words("".join(seg)) > HARD_CAP * 2:  # pathological: blind split
            for k in range(0, len(seg), 400):
                out.append((x + k, min(x + k + 400, len(seg))))
        else:
            out.append((x, y))
    return out


def main():
    manifest, warnings = [], []
    for src in co.content_list():
        path = os.path.join(ROOT, src)
        text = open(path, encoding="utf-8").read()
        lines = text.splitlines(keepends=True)
        total = words(text)
        uid = src.replace("/", "__").removesuffix(".md")
        if total <= MAXW:
            manifest.append({"src": src, "dst": co.dst_path(src),
                             "words": total, "parts": [path],
                             "outs": [os.path.join(WORK, "out", uid, "p01.md")]})
            continue
        # build blocks, greedily pack into parts
        bl = []
        for a, b in blocks(lines):
            if words("".join(lines[a:b])) > HARD_CAP:
                bl.extend(split_big(lines, a, b))
            else:
                bl.append((a, b))
        ranges, cur, curw = [], [bl[0][0]], bl[0][0]  # start of current part
        acc = 0
        pieces = []  # list of (start, end)
        start = 0
        for a, b in bl:
            w = words("".join(lines[a:b]))
            if acc and acc + w > MAXW:
                pieces.append((start, a))
                start, acc = a, w
            else:
                acc += w
        pieces.append((start, len(lines)))
        pdir = os.path.join(WORK, "parts", uid)
        os.makedirs(pdir, exist_ok=True)
        parts_paths = []
        for n, (a, b) in enumerate(pieces, 1):
            pp = os.path.join(pdir, f"p{n:02d}.md")
            open(pp, "w", encoding="utf-8").write("".join(lines[a:b]))
            parts_paths.append(pp)
            pw = words("".join(lines[a:b]))
            if pw > HARD_CAP:
                warnings.append(f"{src} part{n} still large: {pw}w")
        # sanity: parts concatenate back to the original exactly
        joined = "".join("".join(lines[a:b]) for a, b in pieces)
        assert joined == text, f"split is not lossless for {src}"
        manifest.append({"src": src, "dst": co.dst_path(src), "words": total,
                         "parts": parts_paths,
                         "outs": [os.path.join(WORK, "out", uid, f"p{n:02d}.md")
                                  for n in range(1, len(pieces) + 1)]})
    with open(os.path.join(WORK, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    n_parts = sum(len(m["parts"]) for m in manifest)
    print(f"files={len(manifest)} translation units={n_parts}")
    for w in warnings:
        print("WARN:", w)


if __name__ == "__main__":
    main()
