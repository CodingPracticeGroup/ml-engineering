# Mirror all non-md files from content dirs (and root images/) into docs/,
# preserving relative paths so links inside translated pages keep working.
import os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_order as co

ROOT = co.ROOT
DOCS = os.path.join(ROOT, "docs")
CONTENT_DIRS = sorted({p.split("/")[0] for p in co.content_list()
                      if "/" in p}) + ["images"]
SKIP_DIRS = {".git", "docs", "zh_work", "scripts", "__pycache__", "site"}


def main():
    n = 0
    for top in CONTENT_DIRS:
        for dirpath, dirnames, files in os.walk(os.path.join(ROOT, top)):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in files:
                if fn.endswith(".md"):
                    continue
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(src, ROOT)
                dst = os.path.join(DOCS, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    print(f"synced {n} asset files into docs/")


if __name__ == "__main__":
    main()
