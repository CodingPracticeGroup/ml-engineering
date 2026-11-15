# SLURM 性能 {#slurm-performance}

在这里，你会看到关于那些会影响性能的 SLURM 专有设置的讨论。

## srun 的 `--cpus-per-task` 可能需要显式指定 {#sruns---cpus-per-task-may-need-to-be-explicit}

你需要确保由 `srun` 启动的程序能拿到预期数量的 CPU 核心。例如，在典型的 ML 训练程序场景中，每个 GPU 至少需要一个 CPU 核心给驱动它的进程，另外还需要几个核心供 `DataLoader` 使用。之所以需要多个核心，是为了让每个任务都能并行执行。如果你有 8 个 GPU，并且每个 GPU 有 2 个 `DataLoader` 工作进程，那么每个节点至少需要 `3*8=24` 个 CPU 核心。

每个任务的 CPU 数量由 `--cpus-per-task` 定义，该参数传给 `sbatch` 或 `salloc`，最初 `srun` 会继承这一设置。不过，最近这种行为发生了变化：

以下内容引自 `sbatch` 的手册页（manpage）：

> 注意：从 22.05 版本开始，srun 将不再继承通过 salloc 或 sbatch 请求的 --cpus-per-task 值。如果任务需要该值，必须在调用 srun 时再次请求，或者通过 SRUN_CPUS_PER_TASK 环境变量来设置。

这意味着，过去你的 SLURM 脚本可能长这样：

```bash
#SBATCH --cpus-per-task=48
[...]

srun myprogram
```

由 `srun` 启动的程序当时会拿到 48 个 CPU 核心，因为 `srun` 会继承 `sbatch` 或 `salloc` 的 `--cpus-per-task=48` 设置。但根据上面引用的文档，这一行为在 SLURM 22.05 系列的某个时刻发生了变化，并且到 23.x 系列时肯定已经不再成立。

脚注：我在 SLURM@22.05.09 上做过测试，旧行为当时仍然成立，但在 23.x 系列上情况肯定已经不同。所以这一变化可能发生在 22.05 系列的后期版本中。

因此，如果你保持现状不变，现在程序只会拿到 1 个 CPU 核心（除非 `srun` 的默认值已被修改）。

你可以很容易地用 `os.sched_getaffinity(0)` 来测试你的 SLURM 配置是否受到影响，因为它会显示当前进程有资格使用哪些 CPU 核心。因此，用 `len(os.sched_getaffinity(0))` 数出这些核心的数量应该很容易。

下面是测试你是否受影响的方法：
```bash
$ cat test.slurm
#!/bin/bash
#SBATCH --job-name=test-cpu-cores-per-task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48   # 如果你的环境不足 48 个 CPU 核心，请相应调整
#SBATCH --time=0:10:00
#SBATCH --partition=x        # 请根据你的环境改成正确的分区名
#SBATCH --output=%x-%j.out

srun python -c 'import os; print(f"visible cpu cores: {len(os.sched_getaffinity(0))}")'
```

如果你得到
```
visible cpu cores: 48
```
那么你不需要做改动；但如果你得到：
```
visible cpu cores: 1
```
或者其他小于 48 的值，那么你就受到了影响。

要修复这个问题，你需要把 SLURM 脚本改成下面两种形式之一：

```bash
#SBATCH --cpus-per-task=48
[...]

srun --cpus-per-task=48 myprogram
```
或者：
```bash
#SBATCH --cpus-per-task=48
[...]

export SRUN_CPUS_PER_TASK=48
srun myprogram
```

或者用“一次写好、从此不管”的方式把它自动化：
```bash
#SBATCH --cpus-per-task=48
[...]

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun myprogram
```



## 是否启用超线程（Hyper-Threads） {#to-enable-hyper-threads-or-not}

正如在[超线程](users.md#hyper-threads)一节中所解释的，如果你的 CPU 支持超线程（hyper-threading），你应该能够让可用的 CPU 核心数量翻倍，并且对某些工作负载来说，这可能会带来整体更快的性能。

不过，你应该分别在启用和禁用超线程（HT）两种情况下测试性能，比较结果，并选择能带来最佳效果的设置。

案例研究：在 AWS p4 节点上，我发现启用 HT 会让网络吞吐量慢 4 倍。从那以后，在那套特定的配置上，我们一直保持 HT 处于禁用状态。
