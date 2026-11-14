# 加速器基准测试 {#accelerator-benchmarks}

## 如何为加速器跑基准测试 {#how-to-benchmark-accelerators}

### CUDA 基准测试 {#cuda-benchmarks}

关于如何开展 CUDA 基准测试，有几篇写得很好、讲解详尽的文章：

1. [How to Accurately Time CUDA Kernels in PyTorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
2. [How to Benchmark Code on CUDA Devices?](https://salykova.github.io/sgemm-gpu#2-how-to-benchmark-code-on-cuda-devices) - 这一篇与（1）不同：它建议同时固定 GPU 时钟和显存时钟，而（1）只固定 GPU 时钟。

你可以在 [mamf-finder.py](./mamf-finder.py) 中看到这些做法的实际应用（固定时钟除外）。

### 输入数据会影响测得的性能 {#input-data-affects-measured-performance}

内核耗时并不只是张量形状的函数。流经 GEMM（通用矩阵乘）的*数值*会改变芯片的负荷强度，而这又会反馈到时钟频率上。

动态功耗大致正比于时钟频率与发生翻转的晶体管数量的乘积。高熵或“不可预测”的比特模式，每个周期翻转的晶体管比稀疏、已排序或全零的模式更多，因此形状相同的矩阵乘（matmul）会消耗更多功率。一旦功耗接近设定好的功率（或温度、电流）上限，NVIDIA 的 GPU Boost 以及 AMD 的对应机制就会下调 SM 时钟，以便留在允许范围之内——这一机制在[功耗](../README.md#power-consumption)、[散热](../README.md#cooling)以及芯片的 [V-F-T 曲线](../README.md#silicon-lottery)部分已有说明。因此，仅仅是输入不那么“可预测”，就可能让同一个内核跑得更慢，而代码和形状都没有任何改变。Horace He 的 [Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data](https://www.thonking.ai/p/strangely-matrix-multiplications) 通过对比全零输入与随机输入展示了这一效应；[Input-Dependent Power Usage in GPUs](https://ar5iv.labs.arxiv.org/html/2409.18324) 则测量到，在不同输入模式（熵、稀疏性、位相似度、汉明重量）下，GEMM 的功率摆动幅度可达约 38-40%。另一条影响较小的次要路径是非规格化数（denormals/subnormals）：某些算子处理它们时会走更慢的路径，而 bf16/fp16 明显缩小的规格化数范围让这类数更容易出现——参见 NVIDIA 的 [Flush Denormals with Confidence](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/)。

这对做基准测试的实际影响：

1. **给构造输入的随机数发生器（RNG）设种子**（在创建张量之前调用 `torch.manual_seed(...)`）。没有固定的种子时，即使输入的*分布*没有变化，各次运行之间的输入差异也会在测量噪声之上叠加一小份不可控的抖动。
2. **使用符合现实的分布**，而不是全零或其他病态模式。全零会让功率的读数偏低、吞吐量的读数偏高——这正是 Horace 那篇文章所谈的基准测试陷阱。均匀随机的词元 ID（或你的真实工作负载实际喂入的任何数据）能让测得的功率/时钟状态接近生产环境。
3. **当分布固定时，应预期仍会有很小的影响，而不是零。**设种子只消除了一个噪声来源；它不会让两个数值不同、形状相同的张量在完全相同的时钟下运行。

这与[数值可复现性](../../../training/reproducibility/README.md)是两回事，后者通过确定性算法强制得到*相同的结果*。这里的目标是*相同的计时条件*——同样的数据、同样的功耗、同样的时钟。


<a id="maximum-achievable-matmul-flops-finder"></a>

## 最大可达与可持续矩阵乘 FLOPS 查找器 {#maximum-achievable-and-sustainable-matmul-flops-finder}

最大可达矩阵乘 FLOPS（Maximum Achievable Matmul FLOPS，MAMF）基准测试：[mamf-finder.py](./mamf-finder.py) 源自论文 [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) 中的研究。

详细的讨论以及各类加速器的实测数字，见[最大可达与可持续 FLOPS](../README.md#maximum-achievable-and-sustainable-flops)。

一些加速器厂商会公布理论 TFLOPS，但这些数值通常无法达到。因此，当我们尝试优化软件时，没有一个现实的性能标杆可以对照。模型 FLOPS 利用率（Model FLOPS Utilization，MFU）这一指标衡量的是实际达到的 TFLOPS 相对于理论 TFLOPS 的比例。通常做到约 50% MFU 就算是一场胜利。但这并不能告诉我们，距离真正可达的吞吐量还有多远。

这个基准测试会扫描各种尺寸很大的矩阵乘形状，同时报告最高的*可达* TFLOPS（MAMF——boost 短突发下的值）和最高的*可持续* TFLOPS（MSMF——满载下能稳定维持的值）。由于机器学习负载以大型矩阵乘运算为主，MSMF 是对持续训练吞吐量的现实标杆，而 MAMF 是短时突发的上限。这样一来，就不必再使用此前那个以根本达不到的数值为基准的 MFU，而可以改用模型可达矩阵乘 FLOPS 利用率（Model Achievable Matmul FLOPS Utilization，MAMFU）或模型可持续矩阵乘 FLOPS 利用率（Model Sustainable Matmul FLOPS Utilization，MSMFU），真正把负载优化到接近 100%，尤其是对 MSMF 而言——因为“可达”毕竟还不等于“可持续”。这一点非常重要，因为只有这样才能知道何时应该停止优化。

支持的加速器：

- NVIDIA：Volta、Ampere、Hopper、Blackwell、Rubin
- AMD：MI250 及更高型号
- Intel Gaudi2/3
- Intel GPU（A770、A750、B580 等）
- Apple silicon（MPS）

`--search auto`（默认）根据 GPU 的计算单元布局推导测试形状，而 `mamf-finder.py` 只为 NVIDIA 和 AMD 建立了该布局的模型；在其他设备上，请用 `--search grid` 指定一段范围。功率/时钟遥测（其中的 SM 时钟用于证明某次 MAMF 迭代确实运行在 boost 频率下）在 NVIDIA、AMD 和 Gaudi 上都可用（见下文的遥测说明）。[`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) 只能在 NVIDIA 和 AMD 上运行。

重要说明：

- 如果你能找到更好、更高效的方法，把每台新加速器都当作黑盒来探测其最佳矩阵乘 TFLOPS，请把改进连同生成的日志文件一起发 PR 提交给我们。
- 另外，如果你知道这个基准测试需要在某些特殊条件下才能跑出最佳结果，比如某些内核设置之类，请提交 PR 补充这些特殊说明。例如，有人告诉我，在 AMD MI300X 上关闭 numa_balancing 应该会有帮助。
- 由于很大一部分开销来自 HBM IO，如果你使用的是包含两个或更多矩阵乘的融合内核，并且其结果不离开加速器的寄存器，那么性能一定会比本基准测试报告的更快。
- 采样加速器的实际时钟频率也有帮助。如果你的加速器运行在比规格说明所用频率更低的时钟下，那你根本没有机会达到理论 TFLOPS（参见[如何计算理论 TFLOPS](../README.md#how-to-calculate-theoretical-tflops)）。

### MAMF 与 MSMF 的定义方式 {#how-mamf-and-msmf-are-defined}

这两个数值都在同一批候选形状上、以相同的每次迭代工作量测得：先把一个 256MB 缓冲区清零，以把 L2 缓存刷掉，然后重写输出矩阵，再执行矩阵乘。只对矩阵乘本身计时（用设备事件），TFLOPS 等于 `2*M*N*K` 除以这段耗时。

**MAMF**（Maximum *Achievable* Matmul FLOPS，最大*可达*矩阵乘 FLOPS）是短突发在 boost 时钟（GPU 自动升频后的工作频率）下达到的速率。每个形状执行 10 次突发，每次突发包含 20 次背靠背的连续迭代，每次突发之前都空闲 0.25s，让 SM 时钟回升到 boost 频率。一个后台线程每约 0.5ms 采样一次 SM 时钟，每次迭代与其运行期间采样到的最低时钟配对。若该时钟不低于整个运行过程中所见最高时钟的 0.97 倍，这次迭代就算作 boost 迭代。某个形状的 MAMF 是其 5 次最快 boost 迭代的中位数，因此某一次运气好的迭代左右不了结果。在 H200 和 B200 上，“5 次取中位数”与“单独最快的一次”相差最多不超过 0.7%。最终对外报告的 MAMF，是所有达到过 boost 频率的形状中最高的那个形状 MAMF。

**MSMF**（Maximum *Sustainable* Matmul FLOPS，最大*可持续*矩阵乘 FLOPS）是某个形状在不下降的情况下能维持的最高速率，在一段计时窗口内测得，测量前 GPU 已满载运行了足够长的时间，使其时钟已经稳定：

1. **稳定时钟。**用最多 20s 的 8192x8192x8192 BF16 矩阵乘把时钟跑稳，一旦 SM 时钟不再下降就提前停止。空闲的 GPU 会升频、读数偏高，这一步把它降到满载下功率上限所允许的时钟。这里不等待温度稳定：在 H200 上，有一块 GPU 在随后几分钟的测量过程中继续从 63°C 升温到 72°C，但只要没到降频点，升温带来的代价在大多数 GPU 上都不到 1%（参见[散热](../README.md#cooling)）。
2. **排名。**每个候选先跑 0.2s 不计时，再跑 1s 计时。
3. **窗口。**最快的 10 个形状获得完整窗口：先 1s 不计时，再计时 2s，分成 8 个连续子窗口。窗口的 TFLOPS 等于总 FLOPs 除以矩阵乘总耗时。若某个形状最后 4 个子窗口比最前 4 个子窗口慢不超过 1%，就判定它是平稳的。各子窗口之间的离散程度也不得超过 10%：在撞上功率上限时，单个 0.25s 子窗口的读数会有最高约 4% 的抖动，而窗口的均值保持稳定。
4. **K walk**（仅 `--search auto`，见下文解释）。从最快的 4 个平稳布局和最接近正方形的单波（1-wave，下文解释）布局出发，逐步下调 K，每一步都用同样的窗口测量。
5. **锁定。**取最快的平稳形状，或比所有平稳形状都快的那个形状，在 4s 窗口上重新测量，若它在这个窗口上平稳就发布它；否则改试次快的形状，最多试 4 个。如果没有任何形状能在 4s 窗口上保持平稳，就发布在 2s 窗口上平稳的最快形状。如果一个平稳形状都没有，`mamf-finder.py` 会发出警告，并发布最快的那个不平稳形状。

窗口取的是时间，而不是迭代次数，因为固定 100 次迭代，在 B200 上跑快的形状只要 13ms，跑慢的形状则要约 1s，而且在一个形状刚开始跑后的零点几秒内，GPU 仍会保留一部分 boost 频率。在一块 H200 上，0.3s 窗口对 `2816x1536x20480` 的读数是 745–752 TFLOPS，其 4s 锁定窗口为 728–735，10s 的完整运行则是 736–739。功率不参与定义：只要一个形状的吞吐量不下降，它就是可持续的，无论是什么让它维持这个速率。打印在 MSMF 旁边的 `W` 和 `MHz`，是计时窗口内瞬时功率与 SM 时钟的平均值。在 NVIDIA 上，瞬时功率取自 NVML 的 `NVML_FI_DEV_POWER_INSTANT` 字段，因为 NVML 默认的功率读数是对 1s 的平均。

**K walk。**在撞上功率上限时，短突发偏爱的 K 并不是最能持续的 K。更短的 K 能让同一个 `(M,N)` 在相同功率下维持更高的时钟，直到时钟触顶：在 H200 上跑 BF16 时，`2816x1536` 在 K=20480 时可维持 727 TFLOPS，K=10240 时为 741，K=6144 时为 749（1879 MHz，685 W），到 K=4096 又跌回 722——此时时钟已停在 1980 MHz 的上限上，功率则降到了 700 W 上限之下。搜索产生的候选都位于长 K 一侧，所以 K walk 会取最快的 4 个平稳 `(M,N)` 布局，把每个布局的 K 按 K/8（向下取整到 2 的幂；从 K=20480 出发即每次减 2048）逐步下调，最多走 10 步，一旦某一步比本次 walk 中的最好值低超过 1% 就停止。一次 walk 的起点从不超过搜索所用的最大 K（20480）：超过它的形状是为 MAMF 准备的，而更长的 K 只会更不可持续。要取多个布局，是因为两个布局在长 K 下可能看起来一样，到短 K 才分开：在 H200 的 FP8 下，`1280x16896` 和 `1536x2816` 在 K=16384–20480 都能维持约 1290–1305 TFLOPS，但只有 `1536x2816` 在 K=10240 时能升到 1335–1372。只从 1 个布局出发做 walk 时，8 次运行中 5 次找到了它；从 2 个布局出发时，11 次中找到了 10 次。要取 4 个，是因为会出现三个并列：在一次额外的 H200 FP8 运行中，三个布局在长 K 下彼此相差 0.2% 以内，而 `1536x2816` 不在被选中 walk 的两个之中，于是 MSMF 得出 1281，而不是约 1330。即使 4 个也会漏：在 32 次 H200 FP8 运行中的一次里，六个布局在长 K 下都落在 1276–1291 TFLOPS，`1536x2816` 的读数是其中最低的，没有被 walk，MSMF 得出 1289，而不是约 1337。所以，那些最接近正方形的 1 波布局（`msmf_kwalk_waves`=1；“一波”指每个 SM 恰好分到一个线程块（thread-block）分块，见[什么样的形状算快](#what-makes-a-fast-shape)），在 H200 上包括 `1536x2816`，无论排名如何都会被以两种朝向纳入 walk。在另外 32 次 H200 FP8 运行中没有再出现遗漏，而这些额外 walk 的代价在 H200 上约为 10s，在 B200 上则测不出来。在一台 A100 PCIe 上，把“walk 2 个布局加 1 波布局”改成“walk 4 个布局加 1 波布局”，使一次运行从 5:13–5:32 延长到 6:06，而得到的 MAMF 和 MSMF 形状相同。在第 11 次两布局运行中，0.3s 的排名窗口把 `1536x2816` 测得比其通常速率低 2.5%，使它没能进入随后获得完整窗口的 6 个形状，于是两次 walk 都不是从它出发，MSMF 低了 4%。这就是排名窗口取 1s、并让最快的 10 个形状获得完整窗口的原因：在 0.3s 加前 6 名的配置下，另外 32 次 H200 FP8 运行中有 2 次同样漏掉了；而在 1s 加前 10 名的配置下，32 次中一次都没有漏，代价只是每次运行多花约一分钟。

**有多接近。**在一块 H200 和一块 B200 上，各跑 5 次 `--search auto`，得到的 MSMF 分别为 743–746 和 1421–1428 TFLOPS，MAMF 分别为 843–850 和 1682–1705。每个被发布的 MSMF 形状，连续运行 10s、共测两轮，都能维持在其 MSMF 的 0.6% 以内。其中最好的 10s 速率在 H200 上是 750，在 B200 上是 1433。

### 针对特定架构的说明： {#architecture-specific-notes}

运行基准测试前请先遵循以下特殊设置说明，以获得最佳结果：

**MI300x、MI325X 等**：

1. 关闭 numa_balancing 以获得更好的性能：
    ```bash
    sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
    ```
2. 启用 PyTorch TunableOp：
```bash
export PYTORCH_TUNABLEOP_ENABLED=1
```
这会让第一次迭代非常慢，因为它要在 BLAS 库里为遇到的每个 `matmul` 形状搜索最佳 GEMM 算法，但后续操作很可能比基线快得多。参见 [Accelerating models on ROCm using PyTorch TunableOp](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html) 和 [doc](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md)。在 ROCm 10 上，调优一个新形状约需 2 分钟，这会把一次约 2,000 个形状的搜索拖上好几天，所以 `mamf-finder.py` 在搜索期间会暂停调优（默认内核在好形状上与调优后只差约 1–2%，足以用来排名），只对它确认下来的形状在计时之前进行调优。`--tune tunableop_confirm_max=N`（默认 8）限制这类形状的数量。按此设置，在 MI300X 上跑一次 BF16 的 `--search auto`，大约需要 5 分钟搜索加约 16 分钟调优。

在 ROCm 上，`--search auto` 会打印一条说明：它只在一块 MI300X 上、只在 BF16 下对照过穷举网格检验，结果与 16,000 个形状的网格相差在 0.1% 以内。较新的软件栈可能改变的是胜出的形状而非峰值本身：`10240x15360x8192` 从 663.6（ROCm 6.3，torch 2.5.1）降到 625.5（ROCm 10，torch 2.12），而 `12288x9728x8192` 在两套栈上都稳定在约 660，所以升级软件栈之后应当重新搜索，而不是重新计时旧的胜者。选 GPU 请用 `--cuda_device N`，不要用 `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`：在某些虚拟化的 MI300X 主机上，把 GPU0 隐藏之后，torch 就找不到可用的设备了。

**Intel dGPU（A770、A750、B580 等）**

- 按照 Intel Extension for PyTorch 的[安装步骤](https://pytorch-extension.intel.com/installation?platform=gpu)操作。

**NVIDIA、AMD 和 Gaudi 上的遥测：**SM 时钟用于证明某次 MAMF 迭代确实运行在 boost 频率下，功率用于给初步试探的形状排名，两者都打印在每个数字旁边。MSMF 则只凭吞吐量判定。如果缺少遥测软件包（对应地是 `nvidia-ml-py`、`amdsmi`、`habana-pyhlml`），或读不到功率/时钟，`mamf-finder.py` 会拒绝启动。`--telemetry off` 可以照常运行，但此时 MAMF 没有经过 boost 校验。`amdsmi` 的读数已在 MI300X 上确认，而截至本书写作之时，`pyhlml` 的读数尚未在真实硬件上验证过。在 MI300X 上，一个 BF16 内核之内这张卡就会撞上 750 W 上限，空闲启动之后的突发*更慢*而不是更快，所以就这块具体的 GPU 而言，MAMF 不如 MSMF 可靠。

### 使用示例 {#examples-of-usage}

`K` 是归约维度：`(MxK)*(KxN)=(MxN)`。默认 dtype 是 `bfloat16`。`--dtype` 还接受 `float16`、`float32`、`float8_e4m3fn`（NVIDIA 和 AMD MI350X/MI355X 的 fp8）、`float8_e4m3fnuz`（AMD MI300X/MI325X 的 fp8），以及三种输出为 `bfloat16` 的分块缩放格式：`mxfp8`（操作数为 `float8_e4m3fn`，每 32 个元素配一个 `float8_e8m0fnu` 缩放因子）、`mxfp4`（操作数为 fp4 `e2m1`，每 32 个元素配一个 `float8_e8m0fnu` 缩放因子）和 `nvfp4`（操作数为 fp4 `e2m1`，每 16 个元素配一个 `float8_e4m3fn` 缩放因子）。`mxfp8` 和 `mxfp4` 需要支持 MX 的硬件，例如 NVIDIA Blackwell 或 AMD MI350X/MI355X。`nvfp4` 需要 NVIDIA Blackwell，且两种 fp4 格式都要求 `K` 是 32 的倍数。搜索开始前，`mamf-finder.py` 会先用请求的 dtype 跑一个很小的矩阵乘，如果设备或软件栈跑不了，就带着厂商库的错误退出——对于 fp8 dtype，如果另一种 fp8 变体可以运行，它还会指名那一种。每个形状的计时方式见[MAMF 与 MSMF 的定义方式](#how-mamf-and-msmf-are-defined)。

#### 1. Auto search（默认）——搜索形状空间中 GPU 到处能达到的最佳 {#1-auto-search-default-best-the-gpu-can-do-anywhere}

```bash
./mamf-finder.py --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
# 等价写法：./mamf-finder.py --search auto --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

它借助硬件启发式规则找出接近峰值的形状，并报告**两个**摘要数字。想要一个“规格表”式的数字时，它非常合适；但它不针对任何特定模型。[MAMF 与 MSMF 表](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)就是用它产出的。在 H200 和 B200 上，无论 BF16 还是 FP8，每一次 `--search auto` 运行的结果与“对 128,000 个形状做穷举式初探、再走同一套 confirm（即上一节“如何定义 MAMF 与 MSMF”所述排名、窗口与锁定流程）”相比，差距都在 0.6% 以内，而且多数运行比它更高，最多高出 3.6%。在单块 H200 或 B200 上一次运行约需 4 分钟，而在 8 块 GPU 上做那次穷举初探要 17–29 分钟。

- **MAMF**（Maximum *Achievable* Matmul FLOPS，最大*可达*矩阵乘 FLOPS）——boost 时钟下短突发的上限。
- **MSMF**（Maximum *Sustainable* Matmul FLOPS，最大*可持续*矩阵乘 FLOPS）——满载下的稳定速率，与真实训练吞吐量相符。**要为真实模型挑选要用的形状，MSMF 才是该看的数字。**

两者的定义见[如何定义 MAMF 与 MSMF](#how-mamf-and-msmf-are-defined)。一次运行耗时多久取决于 GPU：H200 上约 4 分钟，B200 上 4.5 分钟，A100 PCIe 上 6 分钟，其中包括自适应预热、召回筛选、稳定时钟这一步，以及两次 confirm。在 AMD 上若启用 TunableOp，为每个要确认的形状调参会额外增加约 2 分钟，例如在 MI300X 上一次运行花了 20 分钟（见[针对特定架构的说明](#architecture-specific-notes)）。

`--search auto` 如何选择候选形状（随后两种模式共用后文相同的 confirm 阶段）：

<!---->

1. **波（wave）候选 @ 一组短 K** —— 枚举那些能让线程块分块在所有 SM 上恰好填满整数个完整波（wave）的 `(M,N)` 形状（对 1..16 每个波数分别取最接近正方形/最宽/最高的），并在 K = 1024、2048、3072、4096、8192、12288、14336、16384 和 20480 下测量*每一个*这样的 `(M,N)`。这覆盖了 K 的两端，把波完美盆地整个包住；之所以要在所有这些 K 下测量全部波候选（而不只是排名靠前的前 N 个），就是为了把 H200 的高 K 胜者（`1536x2816x20480`）和 B200 的低 K 胜者（`3072x18944x2048`）都留在 confirm 集合里。若没有 K=2048–4096，B200 上的运行只有第 3 步的种子碰巧落在最佳突发系列附近时才可能达到它，而且同一块 GPU 上 MAMF 会波动 1–2%。
2. **粗粒度 M×N 平面 @ `Kmin` 和 K=3072** —— 在最小的 K 和低 K 的 boost 盆地上跑廉价的网格。后者是必需的：穷举搜索在 B200 上发现的 BF16 MAMF 系列不属于波候选，只能靠它找到。
3. **紧致的局部网格**：围绕按功率排名的初探种子和按原始峰值排名的初探种子（M/N 上步长 256、±4 步，K 上步长 1024、±2 步）——对偏离主轴的峰值做收官打磨，同时不让面向 MSMF 的排名把 MAMF 盆地漏掉。
4. **MAMF 召回筛选** —— 用真实的空闲+突发状态（空闲 50ms 后跑 2 次迭代），廉价地测量原始初探的领先者，加上每个波 `(M,N)` 布局中排名靠前的几个 K，并把筛选出的领先者提升进一份专门的 MAMF 候选短名单。初探之后 GPU 还是热的，所以某次突发若始终没达到 boost 时钟，会在 confirm 前的 0.25s 空闲之后重试：否则在 H200 上，头几个被筛选的形状只会以 1635 MHz 跑突发，而 boost 是 1980 MHz。随后，筛选还会去看搜索所用的最大 K（`max_size`，20480）之外：对最多 8 个布局（`k_edge_layouts`）——即在该 K 下的突发超过其此前被筛选过的所有更短 K 的读数——按增益最陡者优先，再筛选 1.5 倍和 2 倍的 K（上限 `k_edge_max`=65536），一旦突发不再上升就停止。H200 BF16 的 `2816x1536` 在 K=20480 时突发到 828 TFLOPS，在 K=32768 时是 842。初探看不到这一点：初探是在高温下运行的，长 K 的调用降频更多，所以初探读出的是下降曲线。

    共享的 confirm 阶段（`--search auto` 与 `--search grid` 完全相同，区别仅在于 grid 只测量其范围内的形状，因此会跳过 K 边缘筛选和 MSMF 的 K walk）：

5. **候选并集 + 稳定时钟** —— 合并独立排名的 MAMF 短名单和 MSMF 短名单，然后在做任何可持续性测量（`msmf_settle_clock_s`）之前，让芯片满载运行，直到时钟不再下降。若没有这一步，空闲的卡会升频、读数偏高，MSMF 就会取决于这块卡在运行前在做什么。
6. **MSMF confirm**（可持续）：对并集中的每个形状，执行[如何定义 MAMF 与 MSMF](#how-mamf-and-msmf-are-defined)中描述的排名、窗口、K walk 与锁定。并集总会包含 (a) 对每个最接近正方形的低波数形状（w=1..4），在**两种** `(M,N)` 朝向下初探得到的最优 K ——例如 H200 的 `1536×2816` 系列——以及 (b) 初探发现的**最“肥”**的形状（`min(M,N,K)` 最大者，其次体积最大者），它们总能可靠地撞上功率上限。
7. **MAMF confirm**（可达）：对**与 MSMF 相同的并集**重新测量——不用 MSMF 的候选列表，也不按初探时钟做任何预筛。一个在初探期间已经饱和的肥形状，在短暂空闲之后仍能回到 boost 频率（在 B300 上实测）。每次突发**只入队并同步一次**：如果每次起始事件之前都再同步一次，那么取起始事件的瞬间 GPU 恰好处于空闲，内核启动延迟和空闲导致的 DVFS（动态电压与频率调节）重新升频就会被算进被计时的内核头上——依形状不同占 1.6–19%，对访存足迹大的形状最严重，而这会打乱候选的排名。恢复每次迭代的时钟，是把该迭代在 GPU 时间轴上的区间投影到后台采样线程的主机时间轴上，取该区间内的最低时钟。两个摘要数字都选定之后，`mamf-finder.py` 会打印一份**同形状交叉核对**：对每个获胜形状，在两种状态下都测一遍，这样从 boost 到饱和的损耗就能在同一个 GEMM 上直接比较。

每个形状只初探一次，所以一个被多个阶段同时提出的候选不会多花任何开销。

**可复现性——单次运行给出的数字，意义就在于此。**一个对外公布的数字，必须能在同一块 GPU、同样的配置下复现，因此：

<!---->

- **测量时让其他 GPU 保持计算状态。**真实负载会让节点上的每块 GPU 同时运转，而它们共享整板的功率/散热预算——在其他卡空闲时测量，MSMF 会多出一整节点运行时绝不会有的余量。[`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) 在它没有测量的每块 GPU 上跑连续的矩阵乘，把它们顶在自己的功率上限处，同时进行测量：先在 GPU0 上完成完整搜索，得到 MAMF、MSMF 及对应形状；然后轮流把这些形状固定（pin）在其余每块 GPU 上测量，此时包括 GPU0 在内的所有其他 GPU 都在跑矩阵乘。节点的 MAMF 和 MSMF 取的是中位数 GPU：同一节点上的 GPU 芯片之间存在百分之几的差异，中位数刻画的是 GPU 型号本身，而不是某一颗芯片的运气，而且一块坏卡几乎撼动不了它。汇总里还会给出最慢 GPU 的数字——同步训练实际就是按那个速度运行的——以及离散范围。GPU0 的搜索会带上你给出的全部参数，而各轮固定形状的运行带上同样的参数、只是去掉形状选择部分；该脚本显示的是 GPU0 的控制台，把每块 GPU 的日志写到各自的文件，并且全程使用启动它的那个 python，所以要用装着你想要测量的 torch 的那个 python 来启动。在 8 块 H200 上，默认的 `--search auto` 用时约 9 分钟，得到节点 MSMF 755 TFLOPS，最慢的 GPU 为 745；在 8 块 B200 上同样约 9 分钟，得到 1439，最慢的 GPU 为 1421。在这些节点上，单块 GPU 的 MSMF 在两次运行之间的变化最高 1.3%，而同一节点不同 GPU 之间相差最高 4%，所以在说某块 GPU 慢之前，先重跑一遍。
- **单卡：单独运行 `mamf-finder.py`。**在同板的卡都空闲时，MAMF 是最好情况的 boost 短突发，MSMF 是单卡上限。`mamf-finder.py` 在整个 MSMF 测量期间持续采样同板的其他卡，若有卡处于空闲，它会在结果下方说明，并列出每块空闲 GPU 空闲时间的占比。
- **两个摘要数字都不是碰运气的最大值。**MSMF 是保持平稳的那个 4s 窗口的均值，MAMF 是 5 次最快 boost 迭代的中位数；测量时的功率和时钟就打印在数字旁边（`… 985W 1360MHz`）。
- **要复现某个已公布的数字，就用 grid 模式重跑它的那个确切形状**——例如 `--m 9472 --n 6144 --k 12288`。Auto 的职责是在任意位置*寻找*接近峰值的形状；grid 则用于*复现*（或在其中搜索）一个已知范围，同样会执行相同的 MAMF/MSMF confirm。重跑 `auto` 会重新搜索，可能落在另一个（等价的）形状上。

结果在两次运行之间会动多少：这里以两款 GPU 为例展示（并非对每种 GPU 都做过这种测量）。在默认设置下，对一个 8× H200 节点（FP8）和一个 8× B200 节点的每块 GPU 各跑 4 次、8 块同时测量：一块 GPU 的 MSMF 中位变化 0.5–0.6%，最大 2.2%；MAMF 中位变化 0.3–0.9%，最大 1.5%；64 次运行中没有一次错过最佳形状。那次 2.2% 出自某块 H200 的一次运行，其 4s 锁定窗口的读数比同一形状在 K walk 中的窗口低 1.6%，当时 GPU 温度正从 63°C 升到 72°C。如下文实测，同一节点上 GPU 之间的差异，比单块 GPU 两次运行之间的波动更大。

**实测内容：单卡独占 vs. 8 卡并发。**

对两款芯片采用相同流程——先在其余卡空闲的情况下于 GPU0 上跑 5 次 `--search auto`，然后在全部 8 块 GPU 上同时运行同一个脚本共 5 轮：

*B200 bf16（torch 2.14.0+cu130）：*

| 配置                          | MSMF 均值      | MSMF 范围 | MSMF 时钟     | MAMF 均值 |
| :--------------------------- | -------------: | ---------: | ------------: | --------: |
| GPU0 独占 (n=5)              |         1423.8 |       0.5% | 1396–1411 MHz |    1697.6 |
| GPU0，8 卡全忙时 (n=5)       | 1417.2 (−0.5%) |       0.5% | 1387–1396 MHz |    1688.8 |
| 全部 8 卡 × 5 轮 (n=40)      | 1437.5 (+1.0%) |   **4.9%** | 1368–1527 MHz |    1746.7 |

*H200 bf16（torch 2.14.0+cu130）：*

| 配置                          | MSMF 均值     | MSMF 范围 | MSMF 时钟     | MAMF 均值 |
| :--------------------------- | ------------: | ---------: | ------------: | --------: |
| GPU0 独占 (n=5)              |         744.8 |       0.4% | 1835–1873 MHz |     848.4 |
| GPU0，8 卡全忙时 (n=5)       | 743.4 (−0.2%) |       1.1% | 1850–1872 MHz |     847.2 |
| 全部 8 卡 × 5 轮 (n=40)      | 755.5 (+1.4%) |       3.4% | 1850–1938 MHz |     852.1 |

经验教训：在这两个节点上，其他 GPU 同时计算，几乎不影响被测 GPU 自身的数字。GPU0 的 MSMF 在 B200 上只降了 0.5%，在 H200 上只降了 0.2%，MAMF 保持在运行间噪声之内，因为 boost 短突发只消耗约 150–330 W。真正拉开差距的是芯片本身：在 40 次运行中，B200 节点的 MSMF 跨度达 **4.9%**，H200 节点为 3.4%，而且每块 GPU 一轮接一轮地待在这个区间的自己那一小块里（在 B200 上，GPU0 平均 1417，GPU5 平均 1472）。GPU0 恰好是两个节点上最慢的芯片，这就是 8 卡均值在它之上的原因。所以单块 GPU 的 MSMF 刻画的是那颗芯片，而不是 GPU 型号，我们发布的节点数字是中位数 GPU 的。整节点同步训练仍然要等它最慢的那块 GPU，所以它实际得到的是 8 卡区间的下缘。这种并发本身代价多大，取决于整板共享的功率与散热预算，所以要测量整个节点。在[结果表](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)中，`Sib` = `yes` 对应 `mamf-finder-all-gpus.py` 的测量——其余每块 GPU 都在跑连续的矩阵乘——报告的 MAMF 与 MSMF 是中位数 GPU 的；`Sib` = `no` 对应一块 GPU 在其他卡空闲时的测量。

`mamf-finder.py -h` 会列出各项选项。算法自身的调节旋钮——即上面各步骤里用反引号标注的名字（`confirm_reps`、`msmf_settle_clock_s` 等）——都集中放在脚本末尾附近的 `Tuning` 类里，连同各自的默认值和一行用途说明。想试用其他取值，可传入例如 `--tune msmf_settle_clock_s=30 --tune confirm_reps=7`。

#### 阅读输出 {#reading-the-output}

控制台对每个阶段的展示是：一个标题、一行表头、一行会随着形状测量就地更新的数据行；某个阶段结束时，它的获胜数据行会留在屏幕上。在一块 H200 上，`mamf-finder.py --dtype bfloat16 --m_range 1024 4097 1024 --n 1536 2816 --k 4096 8192 20480` 打印如下：

```
Grid search: sweeping 24 shapes (scout 20 iters / 8 warmup), then confirm ...
     #  MxNxK                mean median    max   best     W   MHz
    18  3072x2816x20480     828.4  828.6  831.6  828.4   506  1980  <- best mean

MAMF recall screen: 8 shapes ...
     #  MxNxK                peak   MHz
     1  3072x2816x20480     837.7  1965  <- best peak

MSMF (sustainable) confirm: 10 shapes ...
  MxNxK              TFLOPS     W   MHz
  3072x2816x8192      722.5   681  1612 drop   0.7% spread  4.2%  locked in  <- MSMF

MAMF (achievable) confirm: 10 shapes ...
  MxNxK              TFLOPS     W   MHz  top peaks
  3072x2816x20480     842.3   172  1980  842.8 842.3 842.3 841.9 841.2  <- MAMF

--------------------------------------------------------------------------------

** Results:

Tried 24 shapes => the best outcomes were:
MAMF (max achievable,  boost burst): 842 TFLOPS @ 3072x2816x20480 (MxNxK)  172W 1980MHz
MSMF (max sustainable, saturated):   722 TFLOPS @ 3072x2816x8192 (MxNxK)  681W 1612MHz
note: 7 same-board sibling GPU(s) sat idle during the MSMF measurement, so MSMF reads high.
      Share of the time each was idle: GPU1 100%, GPU2 100%, GPU3 98%, GPU4 94%, GPU5 94%, GPU6 94%, GPU7 94%
      With all siblings idle it is a single-GPU upper bound; for full-node MSMF use mamf-finder-all-gpus.py.
```

MSMF 那一行的数字来自 4s 锁定窗口，发布的摘要数字用的就是它：`drop` 是该窗口后半段比前半段慢的百分比，`spread` 是它的 8 个子窗口之间读数的差异程度。MAMF 行的 `top peaks` 是该形状 5 次最快的 boost 迭代，其中位数就是发布的摘要数字。这两个数字在这里落在了不同的 K 上：K=20480 的短突发最快，而 K=8192 能在功率上限下维持更高的时钟。同一块 GPU 上的 `--search auto` 则在 `1536x2816x6144` 上找到 MSMF 745——这个 K 不在上面这个网格里。

`W`、`MHz` 以及 MSMF 行中的 `C`（GPU 温度），都是该形状运行期间 NVML（或 amdsmi/hlml）的实时采样。在很短的 MAMF 短突发上，NVML 的功率采样可能滞后于内核：在 A100 PCIe 上，空闲之后约 80 ms 的突发会读到接近空闲的瓦数，所以那个 `W` 不是该突发的实际功耗。MAMF 和 MSMF 的摘要数字以整数 TFLOPS 报告；遇 .5 向上取整。每一行，加上每个决定背后的明细（MSMF 的排名、K walk 与锁定数据行、boost 基准、同形状交叉核对），都会写入 `--output_file` 日志（这两个数字如何使用，见 [MAMF 与 MSMF](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)）。

#### 什么样的形状算快 {#what-makes-a-fast-shape}

峰值 GEMM 形状并不神秘；它们遵循论文中给出的配方（[arXiv:2401.14489](https://arxiv.org/abs/2401.14489)）：

1. **张量核对齐** —— M、N、K 取 `128 B / sizeof(dtype)` 个元素的倍数（bf16→64，fp8→128）。
2. **分块量化（tile quantization）** —— 输出能被内核的分块整齐铺满（典型的高效分块是 128×256），所以 M、N 取 256 的倍数是安全的。
3. **波量化（wave quantization）** —— 线程块总数约恰好是 SM 数的整数倍，因而不留下残缺的尾部波（tail wave）。胜出的形状通常 `wave_eff ≥ ~0.99`。
4. **K 只决定算术强度（AI）** —— 它不参与分块/波的计算。K 越大，越偏向计算受限。实测中存在*两类*高 TFLOPS 盆地：高算术强度（大 K）和多波（最小 K、巨大 M×N）——这正是 `auto` 要同时覆盖这两类的原因。

如果你做的是模型设计工作（挑选模型各层将要生成的形状），另见[向量与矩阵尺寸的可整除性](../../../training/performance/README.md#vector-and-matrix-size-divisibility)。

#### 2. Grid search——在*你*的范围内的最佳形状（实践中常见的情形） {#2-grid-search-best-shape-in-your-range-the-practical-case}

当你关心的是一个**特定的子空间**时——比如你的新模型实际会生成的那些形状、某个单一的训练形状，或你想完整测绘的某款加速器的特定区段——就用 `--search grid`（或者直接传入任意 `--m`/`--n`/`--k`/`_*_range` 参数——这本身就意味着 grid）。

Grid 报告的**摘要数字与 auto 完全相同，仍是 MAMF + MSMF 两个**。唯一的区别是候选怎么选：auto 从硬件启发式规则推导候选；grid 遍历*你*给出的 M/N/K 范围，然后执行共享的 confirm 阶段。对模型工作来说，有意义的答案是**你的范围内的 MSMF**——而不是 GPU 的绝对峰值。

```bash
# 你的模型将使用的形状（示例：M 在 2k..8k 之间，N=K=4096）
./mamf-finder.py --m_range 2048 8193 256 --n 4096 --k 4096 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# 单一确切形状（复现已发表的数字）
./mamf-finder.py --m 1024 --n 1024 --k 1024 --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt

# 在粗格点上做 fp8 遍历
./mamf-finder.py --m_range 0 20480 1024 --n_range 0 20480 1024 --k_range 0 20480 1024 \
    --dtype float8_e4m3fn --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

**请给范围留出足够的余地，让它能达到饱和。**MSMF 至少需要一个能在其窗口内维持稳定速率的形状。范围窄/小的网格可能让所有候选都不平稳；这种情况下 `mamf-finder.py` 会发出警告，并发布最快的那个不平稳形状。请扩大 M/N/K（尤其是 `min(M,N,K)`），让能撞上功率上限的形状进入集合。Grid 不会运行 K walk，所以只有当你的 K 范围包含最佳可持续 K 时，它才会被找到。

你可以用 Ctrl-C 中断一次长时间的 grid 运行，仍然会得到目前为止的最佳结果。更细的步长（用 512 / 256 而不是 1024）会让墙上时间（wall time）变成 8× / 64×。想知道某款加速器上哪些形状容易达到峰值，见[向量与矩阵尺寸的可整除性](../../../training/performance/README.md#vector-and-matrix-size-divisibility)。

针对特定架构的设置（MI300X 的 `numa_balancing` / TunableOp、Intel dGPU 的安装）见前文[针对特定架构的说明](#architecture-specific-notes)。


### 结果 {#results}

我已收集到的测量结果，见[最大可达与可持续矩阵乘 FLOPS 对照表](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)。凡是我能用上某款加速器的场合，基准测试是我自己跑的；用不上的场合，则是好心的贡献者们投入自己的时间拿到了这些数字。对此我非常感谢[他们](../../../contributors.md)。

### 如何提交 MAMF/MSMF 条目 {#how-to-report-mamfmsmf-entries}

在有多块 GPU 的节点上，请用 [`mamf-finder-all-gpus.py`](mamf-finder-all-gpus.py) 测量，而不是单独运行 `mamf-finder.py`。训练时节点上的每块 GPU 都在同时计算，共享整板的功率与散热预算，所以在其他卡空闲时测得的 GPU 会拥有繁忙节点绝不可能给出的余量，它的 MSMF 只是单卡上限。同一节点的 GPU 之间也存在芯片间的差异——在上文实测的 8× H200 和 8× B200 节点上，MSMF 相差最高 4%——所以单块 GPU 的数字描述的是那颗芯片，而非 GPU 型号。这个包装脚本会让每块 GPU 都在其他所有 GPU 跑连续矩阵乘的情况下接受测量，并报告中位数 GPU——表中 `Siblings` = `yes` 的那些行就是这么来的——同时给出最慢的 GPU 和离散范围。它还会对以下 GPU 发出警告：MSMF 窗口不平稳的、比最凉的 GPU 热 15°C 及以上的、功耗低于最高者 90% 的：这类 GPU 拉低中位数，原因是节点的散热状况，而不是 GPU 型号。只有一块 GPU 时，它就直接运行 `mamf-finder.py`，得到的就是 `Siblings` = `no` 的行。传给它的参数与你给 `mamf-finder.py` 的完全相同：

```bash
python mamf-finder-all-gpus.py                        # bf16
python mamf-finder-all-gpus.py --dtype float8_e4m3fn  # fp8
```

在 8 块 H200 或 8 块 B200 上，默认的 bf16 运行约需 9 分钟。

如果你测出了比[表格](../README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)更高的 MAMF 或 MSMF——比如用了较新的 `torch`、CUDA 或 ROCm、BLAS 库或驱动——或者你测量了表格里还没有的加速器或 dtype，请通过 [PR](https://github.com/stas00/ml-engineering/pulls) 提交一个更新表格的贡献，或开一个 [Issue](https://github.com/stas00/ml-engineering/issues)。请附上这次运行的完整日志文件：包装脚本的输出目录（脚本旁边的 `results/all-gpus-<timestamp>/`，或由 `OUT_DIR` 指定），含 `summary.txt` 和每块 GPU 的 `gpu<N>.txt`；若是单 GPU 运行，则提供其 `--output_file` 日志。日志记录了软件栈、形状、功率和时钟，正是这些让条目可以核查、可以复现。另外请说明日志里没有体现的任何系统设置，例如已关闭的 `numa_balancing`。同一块 GPU 重跑，数字可波动最多 2.2%，所以小于这个幅度的“提升”是噪声，不是改进。
