#!/usr/bin/env bash
# Build the Chinese mkdocs site and publish it to the gh-pages branch.
set -euo pipefail
cd "$(dirname "$0")/.."

/home/pig/.venv/mkdocs-zh/bin/mkdocs build -d /tmp/mleng-site-zh

BR=gh-pages
WT=/tmp/mleng-ghpages
git worktree remove --force "$WT" 2>/dev/null || true
if git show-ref --verify --quiet refs/heads/$BR; then
  git worktree add -f "$WT" "$BR"
else
  git worktree add --orphan -f "$WT" "$BR"
fi
find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -r /tmp/mleng-site-zh/. "$WT"/
touch "$WT/.nojekyll"
git -C "$WT" add -A
git -C "$WT" commit -m "site: 简体中文站 $(date -u +%F)" --allow-empty
git -C "$WT" push origin "$BR"
git worktree remove --force "$WT"
echo "deployed to origin/$BR"
