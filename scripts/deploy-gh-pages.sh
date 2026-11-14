#!/usr/bin/env bash
# Build the Chinese mkdocs site and publish it to the gh-pages branch
# (via a throwaway clone — no worktree needed for a fresh branch).
set -euo pipefail
cd "$(dirname "$0")/.."

NO_MKDOCS_2_WARNING=1 /home/pig/.venv/mkdocs-zh/bin/mkdocs build -d /tmp/mleng-site-zh
python3 scripts/inject-en-links.py /tmp/mleng-site-zh

WT=/tmp/mleng-ghpages-git
rm -rf "$WT"
mkdir -p "$WT"
cp -r /tmp/mleng-site-zh/. "$WT"/
touch "$WT/.nojekyll"
git -C "$WT" init -q
git -C "$WT" add -A
git -C "$WT" -c user.name="translation-bot" -c user.email="noreply@users.noreply.github.com" \
  commit -q -m "site: 简体中文站 $(date -u +%F)"
git -C "$WT" remote add origin "$(git -C . remote get-url origin)"
git -C "$WT" push -f origin HEAD:refs/heads/gh-pages
echo "deployed to origin/gh-pages ($(git -C "$WT" rev-parse HEAD | cut -c1-8))"
