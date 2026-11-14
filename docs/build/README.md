# 图书构建 {#book-building}

重要提示：这一部分仍是半成品（WIP）——大部分功能已经可用，但样式表还需要一些改进，才能让生成的 PDF 真正美观。预计几周之内可以完成。

本文档假设你是在仓库的根目录下操作。

## 安装要求 {#installation-requirements}

1. 安装图书构建过程中用到的 python 包
```bash
pip install -r build/requirements.txt
```

2. 下载免费版的 [Prince XML](https://www.princexml.com/download/)。它用于构建本书的 PDF 版本。


## 构建 HTML {#build-html}

```bash
make html
```

## 构建 PDF {#build-pdf}

```bash
make pdf
```

它会先构建 html 目标，然后再用它来构建 PDF 版本。

## 构建 EPUB {#build-epub}

```bash
make epub
```

它会先构建 html 目标，然后再用它来构建 EPUB 版本。


## 检查链接与锚点 {#check-links-and-anchors}

若要验证所有本地链接和带锚点的链接都有效，请运行：
```bash
make check-links-local
```

若还想额外检查外部链接，请运行：
```bash
make check-links-all
```
请少用后一条命令，以免因频繁请求服务器而被封禁。


## 移动 md 文件/目录并调整相对链接 {#move-md-filesdirs-and-adjust-relative-links}


例如：`slurm` => `orchestration/slurm`
```bash
src=slurm
dst=orchestration/slurm

mkdir -p orchestration
git mv $src $dst
perl -pi -e "s|$src|$dst|" chapters-md.txt
python build/mdbook/mv-links.py $src $dst
git checkout $dst
make check-links-local

```

## 调整图片尺寸 {#resize-images}

当引用的图片过大时，可以把它们的尺寸缩小一些：

```bash
mogrify -format png -resize 1024x1024\> *png
```
