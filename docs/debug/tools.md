# 调试工具 {#debug-tools}

## git 相关工具 {#git-related-tools}


### 实用别名 {#useful-aliases}

显示当前分支相对远程默认分支所修改过的所有文件的差异：
```bash
alias brdiff="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff origin/\$def_branch..."
```

同样是上面的命令，但忽略空白差异，也就是加上 `--ignore-space-at-eol` 或 `-w`：
```bash
alias brdiff-nows="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff -w origin/\$def_branch..."
```

列出当前分支相对远程默认分支新增或修改的所有文件：
```bash
alias brfiles="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff --name-only origin/\$def_branch..."
```

有了这个列表后，我们就可以自动打开编辑器，只加载这些新增和修改过的文件：
```bash
alias bremacs="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); emacs \$(git diff --name-only origin/\$def_branch...) &"
```


### git-bisect {#git-bisect}

（给自己的提示：这是从 `the-art-of-debugging/methodology.md` 同步过来的，那里才是真正的原始来源）

接下来讨论的方法应该适用于任何支持二分查找（bisect）的修订版本控制系统。在本节讨论中我们将使用 `git bisect`。

`git bisect` 有助于快速找到导致某个特定问题的提交。

使用场景：假设你原本在用 `transformers==4.33.0`，后来你需要一个较新的功能，于是升级到了最前沿的 `transformers@main`，结果你的代码坏了。两个版本之间可能已经有数百个提交，逐个翻检所有提交来找出导致损坏的那个提交会非常困难。下面介绍如何快速找出罪魁祸首是哪个提交。

脚注：HuggingFace Transformers 实际上很少出这类问题，但鉴于它的复杂性和庞大体量，这种情况仍然会发生；不过只要有人报告，问题很快就会修复。由于它是一个非常流行的机器学习库，所以很适合用来作为调试案例。

解决办法：在已知的 good 提交和 bad 提交之间的所有提交上做二分查找，找出那个罪魁祸首提交。

我们要用 2 个 shell 终端：A 和 B。终端 A 用来运行 `git bisect`，终端 B 用来测试你的软件。并没有技术上的理由说不能用一个终端完成这些，但用两个会更轻松。

1. 在终端 A 中获取 git 仓库，并以开发模式（`pip install -e .`）安装到你的 Python 环境中。
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
现在，当你运行应用程序时，将自动使用这个克隆中的代码，而不是你此前从 PyPi、Conda 或其他地方安装的版本。

此外，为简单起见，我们假设所有依赖都已经安装好了。

2. 接下来启动二分查找——在终端 A 中运行：

```bash
git bisect start
```

3. 找出最后一个已知的好（good）提交和第一个已知的坏（bad）提交

`git bisect` 只需要 2 个数据点就能工作。它需要知道一个较早的、已知可以正常工作的提交（`good`），以及一个较晚的、已知会出问题的提交（`bad`）。因此，如果你查看某个给定分支上的提交序列，其中就有 2 个已知的点，以及它们周围许多质量未知的提交：

```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->---------------->----------------> time
```

例如，如果你知道 `transformers==4.33.0` 是好的，而 `transformers@main`（`HEAD`）是坏的，可以访问[发布页面](https://github.com/huggingface/transformers/releases)并搜索 `4.33.0`，找出与标签 `4.33.0` 对应的提交。我们发现它是 SHA 为 [`5a4f340d`](https://github.com/huggingface/transformers/commit/5a4f340df74b42b594aedf60199eea95cdb9bed0) 的那个提交。

脚注：对于给定的仓库，通常前 8 个十六进制字符就足以构成唯一标识，但你也可以使用完整的 40 个字符。


所以现在我们来指定第一个已知的好提交是哪个：
```bash
git bisect good 5a4f340d
```

并且如前所述，我们把 `HEAD`（最新提交）当作坏的那个，这种情况下我们可以直接使用 `HEAD`，而不必先找出它对应的 SHA 字符串：
```bash
git bisect bad HEAD
```

不过，如果你知道问题是在 `4.34.0` 中出现的，可以按上面说明的方法找到它最新的提交，用它来代替 `HEAD`。

现在，找出那个给你惹祸的提交所需的一切都已就绪。

而且在你告诉 `git bisect` 好提交和坏提交之后，它已经切换到了中间的某个提交：

```
...... orig_good ..... .... current .... .... ..... orig_bad ........
------------->--------------->---------------->----------------> time
```

你可以运行 `git log` 查看它切换到了哪个提交。

提醒一下，我们是用 `pip install -e .` 方式安装的该仓库，所以 Python 环境会即时更新为当前提交的代码版本。

4. 好还是坏

下一阶段是告诉 `git bisect` 当前提交是 `good` 还是 `bad`：

为此，在终端 B 中运行一次你的程序。

然后在终端 A 中运行：
```bash
git bisect bad
```
如果它失败了；或者：
```bash
git bisect good
```
如果它成功了。


例如，如果结果是坏的，`git bisect` 会在内部把最后一个提交标记为新的坏提交，并再次把剩余提交对半划分，切换到一个新的当前提交：
```
...... orig_good ..... current .... new_bad .... ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

反过来，如果结果是好的，那么你会看到：
```
...... orig_good ..... .... new_good .... current ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

5. 重复，直到没有剩余的提交

不断重复步骤 4，直到找到有问题的提交。

二分查找结束后，`git bisect` 会告诉你是哪个提交导致了问题。

```
...... orig_good ..... .... last_good first_bad .... .. orig_bad ....
------------->--------------->---------------->----------------> time
```
如果你一直跟着这些提交小示意图看，它对应的就是 `first_bad` 提交。

然后你可以访问 `https://github.com/huggingface/transformers/commit/`，把该提交的 SHA 拼接到这个 URL 后面，页面就会带你到那个提交（例如 `https://github.com/huggingface/transformers/commit/57f44dc4288a3521bd700405ad41e90a4687abc0`），并且该页面还会链接到它来源的 PR。随后你可以在那个 PR 里跟帖寻求帮助。

即使需要搜索的提交有成千上万个，只要你的程序运行时间不太长就行：`2**n` 个提交只需要 `n` 步二分查找，所以 1024 个提交 10 步就能搜完。

如果你的程序非常慢，试着把它缩减成一个更小的程序——理想情况下是一个能非常快地展示问题的小型复现程序。很多时候，把你认为与当前问题无关的大段代码注释掉，就完全足够了。

如果你想查看进度，可以让它显示剩余待检查提交的当前范围：
```bash
git bisect visualize --oneline
```

6. 清理

现在用下面的命令把 git 仓库的克隆恢复到与你开始时相同的状态（很可能是 `HEAD`）：
```bash
git bisect reset
```

并且可能会重新安装这个库没有问题的版本，同时向维护者报告该问题。

有时候，问题源于有意为之的破坏向后兼容性的 API 变更，你可能只需要阅读项目的文档，看看有哪些变化就行了。例如，如果你从 `transformers==2.0.0` 切换到 `transformers==3.0.0`，几乎可以肯定你的代码会坏，因为主版本号的差异通常就是用来引入重大 API 变更的。


7. 可能遇到的问题及其解决办法：

a. 跳过

如果由于某种原因当前提交无法测试——可以用下面的命令跳过它：
```bash
git bisect skip
```
然后 `git bisect` 会继续对剩余的提交做二分查找。

如果提交范围的中间某处某个 API 发生了变化，导致你的程序因为完全不同的原因开始失败，这种做法常常很有帮助。

你也可以尝试做一个适配新 API 的程序变体来代替使用，但这并不总是容易做到。

b. 反转顺序

通常 git 期望 `bad` 位于 `good` 之后。


```
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->--------------->---------------->----------------> time
```

现在，如果按修订版本的排列顺序 `bad` 出现在 `good` 之前，而你想找出第一个修复了既有问题的修订版本——你可以反转 `good` 和 `bad` 的定义——不过让同一组词承担两种相反的逻辑含义会造成混乱，因此建议改用一组新的状态——例如 `fixed` 和 `broken`——做法如下。

```bash
git bisect start --term-new=fixed --term-old=broken
git bisect fixed
git bisect broken 6c94774
```
然后继续用以下命令标记提交：
```bash
git bisect fixed
git bisect broken
```
而不是用：
```bash
git bisect good
git bisect bad
```

c. 复杂情况

有时还会遇到其他复杂情况，比如不同修订版本的依赖不一样，一个版本可能需要 `numpy=1.25`，另一个则需要 `numpy=1.26`。如果依赖包的各个版本之间向后兼容，安装较新的版本应该就够了。但情况并不总是如此。所以有时必须在重新测试程序之前，先重新安装正确的依赖。

有时候会遇到这样的情况：有一段提交实际上是因别的原因坏掉的，这时你可以选择一条不包含那段其他坏提交的 `good...bad` 提交范围，或者按照前面说明的方法，尝试用 `git bisect skip` 跳过那些其他原因的坏提交。
