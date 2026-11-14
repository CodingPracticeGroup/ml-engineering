# Merge translated parts (zh_work/out/<uid>/pNN.md) into docs/<dst>.
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_order as co

ROOT = co.ROOT
DOCS = os.path.join(ROOT, "docs")

manifest = json.load(open(os.path.join(ROOT, "zh_work", "manifest.json")))
missing, empty, done = [], [], []
for u in manifest:
    dst = os.path.join(DOCS, u["dst"])
    texts = []
    for op in u["outs"]:
        if not os.path.exists(op):
            missing.append(op)
            texts.append("")
            continue
        t = open(op, encoding="utf-8").read()
        if len(t.strip()) < 20:
            empty.append(op)
        texts.append(t)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    body = "".join(texts)
    if not body.endswith("\n"):
        body += "\n"
    with open(dst, "w", encoding="utf-8") as f:
        f.write(body)
    done.append((u["dst"], len(u["outs"])))
print(f"merged {len(done)} files")
if missing:
    print("MISSING parts:")
    [print("  " + m) for m in missing]
if empty:
    print("SUSPICIOUSLY SMALL parts:")
    [print("  " + e) for e in empty]
sys.exit(1 if missing else 0)
