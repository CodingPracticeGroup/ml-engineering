# 可复现性 {#reproducibility}

## 在基于随机性的软件中实现确定性 {#achieve-determinism-in-randomness-based-software}

调试时，务必为所有用到的随机数生成器（Random Number Generator，RNG）设置固定的种子，这样每次重新运行时都会得到相同的数据 / 代码路径。

在 GPU 上，仅设置种子还不足以得到逐位可复现（bitwise-reproducible）的结果：cuDNN 和 cuBLAS 可以在不同次运行之间，为同一个算子自由选择不同（但各自都有效）的算法，所以真正确定性的重跑还要求强制它们始终选择相同的、确定性的算法——下面介绍的 `cudnn.benchmark`、`use_deterministic_algorithms()` 和 `CUBLAS_WORKSPACE_CONFIG` 这几项设置处理的就是这个问题。

这里牵涉的系统很多，想全部覆盖并不容易——下面尝试覆盖最常用的那些确定性开关：

```python
import os, random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    if use_seed is not None: # 速度更慢！https://docs.pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        # CUBLAS_WORKSPACE_CONFIG 必须在任何 CUDA 上下文创建之前设置，也就是在
        # 下面的 torch.cuda.* 调用之前——如果别处更早就初始化了 CUDA 就来不及了，
        # 而且是静默地来不及，因此这里用显式的 assert，而不是让它悄无声息地失败
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was already initialized before enforce_reproducibility() got to set "
                "CUBLAS_WORKSPACE_CONFIG - move this call earlier, or export "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 before the process starts"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch 的 RNG
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # 多 gpu——没有 gpu 也可以调用

    return seed
```

说明：

- 用不到的那些 RNG 种子设置调用，把它们注释掉即可——例如你并不直接使用 `numpy` 时。不过，你依赖的某个模块可能在你不知情的情况下在其内部使用了它，所以通常最好全部保留，以求稳妥。

- `CUBLAS_WORKSPACE_CONFIG` 补上了 `use_deterministic_algorithms(True)` 单靠自身无法覆盖的一个缺口：cuBLAS——`torch.mm`/`mv`/`bmm` 及其他基于 `matmul` 的算子背后的库——在 CUDA >= 10.2 上可能是非确定性的，因为它会在同一句柄上，为在并行 CUDA 流中运行的例程动态挑选内部工作区（workspace）内存，因此即便种子相同，结果也可能在不同次运行之间不同。若不设置这个环境变量，`use_deterministic_algorithms(True)` 一碰到 `matmul` 算子就会抛出 `RuntimeError`——这就是需要设置它的信号。它**必须在创建 CUDA 上下文之前设置**，也就是在第一次 CUDA 调用之前；如果进程中别处已经初始化了 CUDA 之后再设置，它会被静默忽略，这就是为什么它在上面 `enforce_reproducibility()` 的最开头、先于 `torch.cuda.*` 调用被设置。

    由于被静默忽略的环境变量很容易被漏掉，上面的代码用 [`torch.cuda.is_initialized()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html) 来做检查：如果到 `enforce_reproducibility()` 运行时，`CUBLAS_WORKSPACE_CONFIG` 尚未设置，而 CUDA 已经被之前的某次调用惰性初始化了（例如某个 `.cuda()` 张量、`torch.cuda.set_device()`，或另一个库先接触了 CUDA），就改为 `raise` 抛出异常。注意，这只能捕获经由 PyTorch 自身的惰性初始化路径完成的初始化——如果某个非 PyTorch 的库先创建了原始 CUDA 上下文，`is_initialized()` 是察觉不到的。因此，尽量在程序中尽可能早地运行 `enforce_reproducibility()`。

    NVIDIA 的 cuBLAS 文档只认可两个可保证确定性的取值，而两者都涉及内存与性能之间的权衡——较小的工作区留给 cuBLAS 的算法选择更少，可能限制性能；较大的工作区则让它有更多余地挑选更快的算法，代价是额外的 GPU 显存：

    - `:16:8` - 确定性的，额外内存最少，但较小的工作区可能限制性能
    - `:4096:8` - 确定性的，代价是约 24MiB 的额外 GPU 显存，但保留更多性能余量（两者中更常用的那个，也是上面使用的取值）

        不存在第三种没有代价的选项——确定性在结构上就与让 cuBLAS 在并发流之间自由挑选最快的算法/工作区布局相冲突，所以把它固定下来总会放弃其中一部分自由。`:4096:8` 的性能损失通常是两者中较小的那个（工作区空间越多，被排除的算法就越少），但“较小”不等于“零”——实际损失取决于工作负载和 `matmul` 的形状，可能从可以忽略到相当显著（有报告称在某些特定工作负载上吞吐量几乎减半）。

        通用格式是 `:[SIZE]:[COUNT]`，你会看到它被串接成多个 `SIZE:COUNT` 对（例如未设置时的默认行为等价于 `CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8`）。这种串接形式来自 cuBLAS 内部的工作区内存池配置，其中每一对最初描述一个独立的桶大小/数量，使内存池能高效地服务大小不同的工作区请求。但 PyTorch 只是把它找到的每个 `SIZE * COUNT` 对加总成一个单一的总字节数，然后把这一块固定大小的缓冲区交给 cuBLAS——所以 `:4096:2:16:8` 仅仅意味着总共 `2*4096 + 8*16` KiB，串接更多对只改变*你加总的 KiB 数量*，并不改变确定性行为。实际上你不需要自己串接任何东西：从上面 NVIDIA 认可的两个单对取值中选一个即可，或者设置 `:0:0` 强制 cuBLAS 完全不使用任何工作区（不推荐——这是限制最严、速度最慢的选项）。

- `torch.use_deterministic_algorithms(True)` 使 cuDNN 卷积具有确定性，并且它还额外覆盖 PyTorch 中的数百个其他算子（`index_add_`、`scatter_`、插值等）。它不会只是默默地尽力而为，而是一碰到没有确定性实现的算子就抛出 `RuntimeError`，因此非确定性无法悄无声息地混进来。

- `torch.backends.cudnn.benchmark = False` 是另一个独立的、与之互补的开关，用来阻止 cuDNN 在第一次遇到每个新的输入形状时给几种卷积算法计时，并缓存其中最快的那个。这个跑基准测试的步骤本身就是运行间差异的一个来源——哪种算法在计时竞赛中“胜出”，可能取决于偶发的系统噪声——且与它最终选定的算法是否具备确定性无关。

> **PyTorch 版本说明**：`torch.use_deterministic_algorithms()` 需要 **PyTorch >= 1.8**。在 PyTorch < 1.8 上，退回使用作用范围更窄的 `torch.backends.cudnn.deterministic = True`，它只让 cuDNN 卷积具有确定性，而对其余部分（`matmul`、`index_add_`、`scatter_` 等）不做处理——`CUBLAS_WORKSPACE_CONFIG` 与 PyTorch 版本无关，仍然适用：
> ```python
> if use_seed:
>     os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
>     torch.backends.cudnn.benchmark     = False
>     torch.backends.cudnn.deterministic = True # PyTorch < 1.8 回退方案——仅 cuDNN 卷积
> ```

资源：

- [NVIDIA cuBLAS 文档 - 可复现性](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility) - 关于 cuBLAS 非确定性以及被认可的 `CUBLAS_WORKSPACE_CONFIG` 取值的权威说明
- [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) - 受影响算子的完整列表以及 `warn_only` 的工作方式
- [PyTorch CUDA 环境变量参考](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html) - `CUBLAS_WORKSPACE_CONFIG` 的确切 `:[SIZE]:[COUNT]` 格式规范
- [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html) - 把种子设置、`cudnn.deterministic` 和 `cudnn.benchmark` 与以上内容串在一起，给出完整图景

如果你的代码除了 CPU/CUDA 之外还涉及其他加速器后端或框架——例如 NPU、XPU 或 TensorFlow——也要为它们的 RNG 设置种子：
```python
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

当你为了解决某个问题而反复重跑同一段代码时，在代码开头用下面的方式设置一个特定的种子：
```
enforce_reproducibility(42)
```

如上所述，**这仅用于调试**：它会启用 `torch` 中若干有助于确定性的标志，但可能拖慢速度，所以你不希望把它用于生产环境。

不过，你可以改为这样调用来用于生产环境：
```
enforce_reproducibility()
```
也就是说，不传显式的种子。然后它会挑选一个随机种子并把它记入日志！这样，如果生产中出了什么问题，你现在就能复现问题发生时的那个 RNG 状态。而且这次没有性能损失，因为只有在你显式提供种子时，那些确定性设置（`cudnn.benchmark` / `use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG`）才会被设置。假设日志输出了：
```
Using seed: 1234
```
你只需要把代码改成：
```
enforce_reproducibility(1234)
```
就会得到完全相同的 RNG 设置。

如前所述，一个系统中可能还牵涉许多其他 RNG——例如，如果你希望 `DataLoader` 以相同的顺序输入数据，就[需要把它的种子也设置好](https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader)。

另见调试一章中的[不同设备上的浮点数运算差异](../../debug/pytorch.md#floating-point-math-discrepancies-on-different-devices)。

## 复现软件与系统环境 {#reproduce-the-software-and-system-environment}

当发现结果存在某种差异时——例如质量或吞吐量上的差异——这种方法很有用。

思路是记录启动一次训练（或推理）所用环境的关键组件，这样如果以后需要原样精确复现，就能做到。

由于使用的系统和组件种类繁多，不可能规定一种永远奏效的做法。所以我们来讨论一个可行的配方，你可以再把它调整到自己特定的环境中。

把下面这些添加到你的 Slurm 启动脚本（或你用于启动训练的其他任何方式）中——这是一个 Bash 脚本：

```bash
SAVE_DIR=/tmp # 编辑为真实的持久化路径
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. modules（输出到 stderr）（如果你不使用 lmod 或类似的 modules 实现，请删除）
module list 2> $REPRO_DIR/modules.txt
# 2. shell 环境变量
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. pip（这也包含开发模式安装的 SHA）
pip freeze > $REPRO_DIR/requirements.txt
# 4. 安装进 conda 的 git 克隆中未提交的 diff
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

如你所见，这个配方是在 SLURM 环境中使用的，因此每次新的训练都会转储与该 SLURM 作业相对应的环境。但你可以把它调整为适用于任何其他环境。

下面详细展开配方中的每一步：

1. 我们保存加载了哪些 `modules`，例如在云集群/HPC 设置中，你很可能就是用这种方式加载 CUDA 和 cuDNN 库的。

    如果你不使用 `modules`，那就删掉那一项

2. 我们转储 shell 环境变量。这可能至关重要，因为在某些环境中，仅仅一个 `LD_PRELOAD` 或 `LD_LIBRARY_PATH` 环境变量就可能对性能产生巨大影响

3. 然后我们转储 python 环境中的软件包及其版本——这对 `conda`、`venv` 等任何 python 虚拟环境都应该有效，甚至不使用虚拟环境也可以。

    如果你用的是 `uv` 而不是 `pip`，改用 `uv pip freeze`，它会快得多。

4. 如果你用 `pip install -e .` 做开发模式安装（devel install），除了安装来源那个 git 克隆仓库的 git SHA 之外，它对该仓库一无所知。但问题在于，你很可能在本地修改过文件，而这时 `pip freeze` 会漏掉那些改动。所以这一步会遍历所有未安装到 conda 环境中的软件包（我们通过查看 `site-packages/*.dist-info/direct_url.json` 来找到它们），并对其中每个存在未提交改动的软件包，把它的 `git diff` 保存到 `$REPRO_DIR/<package>.diff`。

若要保存 `apt` 软件包，添加：
```bash
apt list --installed > $REPRO_DIR/apt-packages.txt
```

如果使用 `conda`，另一个有用的工具是 [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)，它能帮你找出两个 conda 环境之间的确切差异。

说个亲身经历：我和一位同事在一个云集群上运行完全相同的代码，得到的训练 TFLOPS 却相差很大——真的是从同一个共享目录启动同一个 Slurm 脚本。我们先用 [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) 比较了双方的 conda 环境，发现了一些差异——我安装了和她完全相同的软件包以匹配她的环境，但依然显示出巨大的性能差距。然后我们比较了 `printenv` 的输出，发现我设置了 `LD_PRELOAD`，而她没有——这造成了巨大的差别，因为这家云服务商要求把多个环境变量设置为自定义路径，才能把他们的硬件性能发挥到极致。
