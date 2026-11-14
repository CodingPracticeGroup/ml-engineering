# Shared definition of the book content list and its order (used by
# make-units.py and gen-mkdocs.py). Paths are repo-relative source paths.
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# extra files that exist in the repo but are not listed in chapters-md.txt
DEBUG_EXTRA = [
    "debug/nccl-performance-debug.md",
    "debug/torch-distributed-hanging-solutions.md",
    "debug/underflow_overflow.md",
    "debug/make-tiny-models-tokenizers-datasets.md",
]

SECTION_TITLES = {
    "insights": "洞见：算力与云服务",
    "compute": "计算硬件",
    "storage": "存储",
    "network": "网络",
    "orchestration": "编排与调度",
    "training": "训练",
    "model-parallelism": "训练",
    "inference": "推理",
    "debug": "调试与排障",
    "testing": "测试",
    "resources": "资源汇总",
    "contributors": "贡献者",
    "courses": "课程：设计并运行一次大模型训练",
    "build": "附录：本书的构建方式",
}


def content_list():
    """Ordered list of source md paths (root README.md first)."""
    with open(os.path.join(ROOT, "chapters-md.txt"), encoding="utf-8") as f:
        chapters = [ln.strip() for ln in f if ln.strip()]
    out = []
    for path in chapters:
        if path == "debug/tools.md":
            out.append(path)
            out.extend(DEBUG_EXTRA)
            continue
        if path == "training/model-parallelism/README.md":
            out.append("model-parallelism/README.md")
        out.append(path)
    return out  # root README.md is the first entry per chapters-md.txt


def dst_path(src):
    """Map a source md path to its path under docs/."""
    if src == "README.md":
        return "index.md"
    return src


def sections(ordered_paths):
    """Group ordered source paths into ordered (section_title, [dst_paths])."""
    sections_map, order = {}, []
    for src in ordered_paths:
        if src == "README.md":
            continue  # home page handled separately
        top = src.split("/")[0]
        if top.endswith(".md"):
            top = top[:-3]
        title = SECTION_TITLES[top]
        if title not in sections_map:
            sections_map[title] = []
            order.append(title)
        sections_map[title].append(dst_path(src))
    return [(t, sections_map[t]) for t in order]
