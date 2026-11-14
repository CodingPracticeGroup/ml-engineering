# Generate mkdocs.yml (Chinese site config + full nav) from the content order.
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_order as co

CONFIG = """\
site_name: 机器学习工程（ML Engineering）· 简体中文版
site_description: 《Machine Learning Engineering》(Stas Bekman) 简体中文翻译，面向计算机专业本科低年级读者
site_url: https://codingpracticegroup.github.io/ml-engineering/
repo_url: https://github.com/CodingPracticeGroup/ml-engineering
docs_dir: docs
use_directory_urls: false

theme:
  name: material
  language: zh
  palette:
    - scheme: default
      primary: indigo
      toggle: {icon: material/weather-night, name: 切换深色模式}
    - scheme: slate
      primary: black
      toggle: {icon: material/weather-sunny, name: 切换浅色模式}
  features:
    - navigation.sections
    - navigation.top
    - navigation.footer
    - toc.follow
    - content.code.copy

plugins:
  - search:
      lang: zh

markdown_extensions:
  - toc:
      permalink: true
  - attr_list
  - admonition
  - footnotes
  - md_in_html
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - pymdownx.superfences
  - pymdownx.details
  - pymdownx.tabbed:
      alternate_style: true
"""


def main():
    ordered = co.content_list()
    # only include pages that exist under docs/ (preview-friendly nav)
    ordered = [p for p in ordered
               if os.path.exists(os.path.join(co.ROOT, "docs", co.dst_path(p)))]
    # bare path nav entries: mkdocs takes each page title from its H1
    nav = ["- index.md"]
    for title, dsts in co.sections(ordered):
        if not dsts:
            continue
        nav.append(f"- {title}:")
        nav.extend(f"    - {d}" for d in dsts)
    with open(os.path.join(co.ROOT, "mkdocs.yml"), "w", encoding="utf-8") as f:
        f.write(CONFIG + "\nnav:\n" + "\n".join(nav) + "\n")
    print("wrote mkdocs.yml")


if __name__ == "__main__":
    main()
