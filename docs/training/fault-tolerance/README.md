# 故障容错 {#fault-tolerance}

无论机器学习训练硬件是你自有的，还是按小时租用的，在这个不断提速的 ML 领域，及时完成训练都至关重要。因此，假如你睡觉时某块 GPU 出了故障，或者存放检查点的存储用满了空间，导致训练崩溃，那么一觉醒来你会发现已经白白损失了很多训练小时。

由于 ML 硬件的成本高得令人望而却步，很难像在 Web 服务中那样提供冗余的故障切换方案。不过，只需要几条简单的做法，就能让你的训练具备故障容错能力。

由于大多数正式的训练作业都在 SLURM 环境中运行，下文会多次提到它，但本章的大部分经验同样适用于其他任何训练环境。

## 始终计划准备比所需更多的节点 {#always-plan-to-have-more-nodes-than-needed}

GPU 设备的现实情况是：它们容易出故障。有时只是过热后关机，但还能恢复；有时则是彻底损坏，需要更换。

随着同一批节点使用几周或几个月，情况会逐渐好转，因为表现差的机器会被陆续替换掉。但如果你有幸拿到新到货的一批 GPU，尤其是在这项技术刚刚问世时拿到的早期 GPU，就要预期其中相当一部分会出故障。

因此，如果你需要 64 个节点来做训练，一定要确保手头有几个备用节点，并且要弄清楚：在手头备用节点不够用时，你能多快替换掉故障节点。

很难准确预测冗余比例应该是多少，但 5-10% 并不算离谱。你越是赶着按时完成训练，安全余量就应当越高。

有了备用节点之后，还要验证你的 SLURM 环境会自动把有问题的节点从可用节点池中摘除，从而自动用好的节点替换坏节点。

如果你用的是非 SLURM 调度器，也要验证它同样能做到无人值守地替换坏节点。

此外你至少还需要一个额外的节点，用来运行各种预防性看门狗程序（watchdog，本章后面会讨论），可能还要用它承担检查点转储（offload）和清理作业。

## 原地重建进程组 {#rebuilding-the-process-group-in-place}

本章其余部分对 rank（分布式训练中每个参与进程的编号，这里指发生故障的进程）故障的处理方式是：进程退出，让排在队列中的下一个作业在一组全新分配的资源上启动。从 PyTorch 2.14 开始，通信器本身可以在原地重建，因此存活的 rank 能够保留其已经预热的进程状态。这条路径在 2.14 中仍是实验性的：你需要在 `init_process_group(..., enable_reconfigure=True)` 中显式启用，而且相关调用（`dist._get_reconfigure_handle`、`dist._reconfigure`）都带有下划线前缀，文档也注明它们将来可能变化。在这些接口稳定之前，请继续使用备用节点加作业数组作为故障恢复手段。


## 排队多个训练作业 {#queue-up-multiple-training-jobs}

下一个关键步骤是确保训练崩溃后，有一个新作业已经排好队接替前一个作业。

因此，启动训练时，不要只用：
```bash
sbatch train.slurm
```

你应该把它换成：

```bash
sbatch --array=1-10%1 train.slurm
```

这告诉 SLURM 预订一个由 10 个作业组成的作业数组（job array），只要其中一个作业正常完成或者崩溃，SLURM 就会立刻调度下一个。

脚注：`--array=1-10%1` 中的 `%1` 告诉 SLURM 串行启动这个作业数组——一次只跑一个作业。

如果你在没有这样安排的情况下已经启动了训练，也不必中止当前作业，用 `--dependency` 参数就能轻松补救：
```bash
sbatch --array=1-10%1 --dependency=CURRENTLY_RUNNING_JOB_ID train.slurm
```
假如你启动的作业看起来像这样：

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-64]
```
你会注意到当前作业的 `JOBID=87`，现在你可以把它用在这里：
```bash
sbatch --array=1-10%1 --dependency=87 train.slurm
```
之后新的状态就会显示为：
```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-64]
  88_[1-10%1]   prod    my-training-10b  PENDING       0:00 1-16:00:00   64                    N/A (Dependency)
```
你可以看到，一个由 10 个作业组成的数组（`88_[1-10%1]`）被追加到了后面，会在当前作业（`87`）完成或失败后立即启动。

诚然，如果导致崩溃的条件依然存在，后续的作业同样会失败。例如，如果存储设备满了，重启多少次也无法让训练继续。我们马上就会讨论如何避免这种情况。

但由于训练崩溃的主要原因是 GPU 故障，确保故障节点被自动摘除、新作业在一组新节点上启动，就能让训练从崩溃中平滑恢复。

按 SLURM 的术语来说，被摘除的节点会被赋予一个名为 `drained` 的新状态。下面是一个假想 SLURM 集群的示例：

```bash
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
prod*       up   infinite       4  drain node-[0-3]
prod*       up   infinite      47  alloc node-[4-51]
prod*       up   infinite      23   idle node-[52-73]
```

这里我们有 47 个节点正在被使用（`alloc`），23 个可用（`idle`），4 个不可用（`drained`）。

系统管理员应当定期检查这些 drained 节点，修复或更换它们，然后把它们的状态改回 `idle`，使其重新可用。

另一种做法是通过 `--dependency` 把作业串成菊花链（daisy chain），如[这里](../../orchestration/slurm/users.md#request-allocation-via-dependency)所述。这两种做法也可以结合起来使用。

如何知道作业数组或菊花链什么时候不该继续运行呢——通常，如果训练循环知道作业已经完成，就会立即退出。你还可以加入像[终止开关](#kill-switch)这样的功能，用它来阻止作业数组继续运行就更简单了。


## 优先使用固定的加速卡分配，而非动态分配 {#preferring-fixed-accelerator-allocations-to-dynamic-ones}

通常，拿到一批新的加速卡节点时——尤其是刚面世不久的新款加速卡——会有大量加速卡出故障，使大语言模型（LLM）训练变得相当棘手。新款加速卡早期的故障率可高达 10%，到了后期故障率仍然相当高。要记住，如果你有 8 张加速卡，哪怕只有 1 张坏了，从训练程序的角度看也等同于 8 张全坏了。

如果你使用固定的节点分配，几个月后，劣质加速卡就会被淘汰，剩下出故障的应该很少。故障仍会发生，但会成为罕见事件。

要确保服务商在加速卡损坏时给你换新卡，而不是等它们冷却一下（字面意义上的冷却）之后又把同一批卡还给你。例如，可以看看如何跟踪 [NVIDIA GPU 的 UUID](../../compute/accelerator/nvidia/debug.md#how-to-detect-if-you-get-the-same-broken-node-again-and-again)。这类间歇性故障在高负载下很可能复发，所以你希望它们被真正替换掉。

如果你使用动态分配，即使新款加速卡已经发布一年，也要预期会有很多故障卡，因为你拿到的往往是其他用户挑剩的节点。诚然，有些云服务商在勤于更换坏硬件方面比其他家做得更好，问题在于有很多加速卡并不是彻底坏死的：当有人丢弃一个坏节点后，检查它的技术人员试机时可能看不出任何问题。而如果用户只是释放了节点却没有报告它坏了，云服务商在把节点交给下一位用户之前又没有复检它是否完好，那么拿到坏节点的概率就非常高。



## 频繁保存检查点 {#frequent-checkpoint-saving}

训练作业一旦失败，就可能损失好几个小时的训练。频繁保存检查点可以缓解这个问题。训练恢复时会从最后一次保存的检查点继续。如果故障发生在距离上一次保存检查点 12 个小时之后，就损失了 12 小时的训练，需要重新跑完。如果训练用到上百块 GPU，这会非常昂贵。

理论上，你可以每 10 分钟保存一次检查点，这样最多只会损失 10 分钟的训练时间，但这同样会大大拖慢到达终点线的进度，因为大模型保存起来并不快，一旦保存时间开始成为训练的瓶颈，这种做法就适得其反了。

取决于你采用的检查点保存方法和 IO 存储分区的速度，保存一个大模型可能需要几十秒到几分钟。因此，最佳的保存频率处在两者之间的某个位置。

这笔账很好算——测量保存一次检查点所需的时间，乘以你希望保存的次数，看看检查点保存会给总训练时间增加多少额外的延迟。

使用案例：在训练 BLOOM-176B 时，我们有一个极快的 GPFS over NVME 文件系统，用 384 个进程并发写入一个 2.3TB 的检查点只需要 40 秒。我们大约每 3 小时保存一次检查点。由于训练进行了约 3 个月，总共保存了约 720 个检查点（`90 days * 24h / 3h`）——也就是说光是保存检查点就多花了 8 个小时（`720 times * 40 secs / 3600 secs`）——约占总训练时间的 0.37%（`8h / (90 days * 24 hours)`）。现在假设 IO 慢 5 倍——除非购买高级 IO 服务，这在云上并不少见——那就会占到训练时间的 2%，这就相当可观了。

脚注：如果你没有大容量本地存储，必须把检查点转储到云端，请确保把最近的 2 个检查点保留在本地，以便快速恢复。之所以是 2 个而不是 1 个，是因为有可能最后一个检查点损坏了，或者崩溃恰好发生在保存它的过程中、没有保存完整。

虽然这种方法会给训练引入开销，但拥有训练检查点极为有用。因为一旦出现发散，它们能让你回滚到许多步之前；它们也有助于分析各种事件。而且如今的许多训练正在从训练中只做单一损失测量的评估（这种评估提供的有用信号很少），转向在训练期间对每个检查点应用多个基准测试的完整数据集评估。后者可以在额外的节点上完成，不会因训练中评估而拖慢训练。


## 基于多副本的故障容错 {#multi-replica-based-fault-tolerance}

还有一种处理加速卡崩溃的方法，完全不涉及保存检查点。这种方法只有在训练至少使用了两个模型副本时才可行。

请先了解各种[模型并行](../model-parallelism/README.md)技术，以便跟上下文。

- 如果使用的是某种变体的 3D 模型并行，也就是张量并行（TP）和/或流水线并行（PP）和/或数据并行（DP），那么副本数量等于 DP 并行度。
- 如果使用的是混合式 ZeRO-DP 并行，那么副本数量等于混合副本的数量。

例如，假设你的训练设置使用 TP=4、PP=2、DP=2 的 3D 并行——那么你有 2 个副本，每个副本使用 8 块 GPU，分别占用 `node0` 和 `node1`（TP=4、PP=2 ⇒ `4*2=8`）——实际上，每个副本恰好用掉一整台 8 GPU 的节点。

此外，你还有一个备用节点 `node2`，它的 8 块 GPU 处于空闲状态，但随时可以在一瞬间投入使用。

现在，假设训练过程中 `node0.gpu0` 出了故障。由于第 2 个副本的数据完好，你就切换到备用的 8 GPU 节点，通过 RDMA 把第 2 个副本各 GPU 上的数据复制过来，然后就能从中断处继续训练。这是非常简化的说明，因为要根据故障发生在迭代循环的哪一个阶段来确定恢复方案，其中有很多细微之处。换句话说，这需要实现一套复杂的算法。

当然，在大规模训练中，你更可能有一百个正在工作的活动节点，外加为数不多的几个备用节点。

这种方法优于基于文件系统的检查点保存，因为你最多只损失一次迭代，而基于文件系统的检查点保存可能让你损失成百上千次迭代。

最接近这一思路的开源库是 [`torchft`](https://github.com/meta-pytorch/torchft)（Meta / PyTorch）。它把每个训练步骤视为副本组之间的故障边界，通过一个 lighthouse/quorum（仲裁）服务协调各方的健康状态，并且可以从一个健康的对等副本实时传输状态来“治愈”出故障的组，而不用回滚到文件系统检查点。它开箱即支持容错的 DDP 和 HSDP，而 [`torchtitan`](https://github.com/pytorch/torchtitan) 将其接入了端到端的 HSDP 训练循环——可参考它的 [`torchft` 实验文档](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/torchft)，以及 [PyTorch 博客关于在合成故障下无检查点运行的文章](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/)。它的故障域是整个副本组而不是单块 GPU，恢复路径是借助 torchft 的检查点传输实现对等恢复，而不是上文“通过 RDMA 复制到专用备用节点”的思路——目标相同（最多损失一个步骤、继续训练），机制不同。

相关的研究系统则从其他角度解决同样的问题。[`Oobleck`](https://github.com/SymbioticLab/Oobleck)（SOSP'23）保留逻辑上等价的流水线副本，故障后通过重新实例化预先计算好的流水线模板来恢复，同样不需要从完整检查点重启。更早的抢占式实例（spot instance）工作，如 [Bamboo](https://www.usenix.org/system/files/nsdi23-thorpe.pdf)（把冗余计算调度进流水线空隙；NSDI'23）和 [Varuna](https://github.com/microsoft/varuna)（弹性的 PP+DP 重配置），也属于同一类；这些代码库如今大多已停更，主要作为设计参考。此外也有商业基础设施层，能在不修改训练脚本的情况下把故障 GPU 迁移到备用卡上——这是一种不同的权衡（故障域按单块 GPU 计、位于框架之外），而不是又一个 torchft 的仿制品。





## 终止开关 {#kill-switch}

在许多 SLURM 环境中，用户没有 `sudo` 权限。如果一个用户启动了训练就去睡觉了，随后发现了问题，其他用户却没法方便地停止训练并重新启动。

在 BLOOM-176B 训练期间我们遇到的正是这种情况，于是实现了一个终止开关（kill switch）来处理它。机制非常简单：训练循环在开始新的迭代之前会轮询检查某个特定文件是否出现，如果该文件存在，程序就保存检查点并退出，从而让启动上一次训练的那个用户以外的人也能做出修改并重新启动训练。我们还在 `main` 的最开头增加了一次轮询，这样如果那位去睡觉的用户排了一个很长的作业数组，可以让每个作业一启动就迅速退出，从而快速把它们“烧”完。

这一点在[这里](../../orchestration/slurm/users.md#overcoming-the-lack-of-group-slurm-job-ownership)也有讨论。

这个功能有助于把浪费掉的训练时间降到最少。

## 保存开关 {#save-switch}

既然提到了终止开关，不妨也顺便提一下它的“近亲”——保存开关。与终止开关类似，保存开关是前者的一个变体：训练循环如果发现出现了保存开关文件，不会停止训练，而是保存一个检查点，然后继续训练。它还会自动把保存开关从文件系统中删除，以免之后每次迭代都意外地保存一个检查点。

这个功能对盯着训练图表看的人非常有用。如果有人在训练损失或其他训练指标中看到了有趣的或关键的情况，就可以迅速请求训练程序保存当前这个感兴趣的检查点，以便日后随时复现当前的情况。

这个功能的主要用途是观察训练损失尖峰和发散。

## 预防 {#prevention}

避免损失训练时间最简单的方法，就是防止某些类型的问题发生。虽然我们无法阻止 GPU 出故障——除了确保提供足够的散热之外——但完全可以确保剩余磁盘空间足够支撑接下来几天的训练。这通常通过运行定时看门狗来实现：它们监控各种资源，在问题真正发生很久之前就提醒运维人员注意可能出现的问题。

### 定时看门狗 {#scheduled-watchdogs}

在讨论各种看门狗之前，关键是你必须有一种能够运行定时作业的机制。在 Unix 世界里，这是通过 [`crontab` 机制](https://en.wikipedia.org/wiki/Cron)实现的。

下面是每隔一小时启动一次 `~/bin/watch-fs.sh` 的示例：
```
0 * * * * ~/bin/watch-fs.sh
```
上面的链接解释了如何把 crontab 作业配置为其他各种运行频率。

设置 crontab 用 `crontab -e`，查看已经调度了哪些作业用 `crontab -l`。

我不在这里展开细节，是因为许多 SLURM 环境并不开放 `crontab` 机制的使用权限，因此必须用其他办法来调度作业。

[Crontab 模拟](../../orchestration/slurm/users.md#crontab-emulation)一节讨论了如何实现类 crontab 的 SLURM 模拟，以及[自我延续的 SLURM 作业](../../orchestration/slurm/users.md#self-perpetuating-slurm-jobs)。


### 通知机制 {#notification-facility}

接下来你需要一种或多种通知机制。

最简单的是用电子邮件发送告警。要用这个办法，你需要确保能从 SLURM 作业里发出邮件。如果环境还不具备这个条件，可以向系统管理员提出这个需求，或者使用外部的 SMTP 邮件服务器服务商。

除了邮件，你大概还能配置其他通知方式，比如短信告警，和/或如果你用 Slack，就向你指定的频道发送 Slack 通知。

一旦你弄清楚了如何调度看门狗，并且有了可用的通知机制，下面就来讨论几个关键的看门狗。

### 检查作业是否在运行的看门狗 {#is-job-running-watchdog}

最显而易见的看门狗，是检查是否有一个训练 SLURM 作业正在运行、或者是否还有作业被调度等待运行。

下面是 BLOOM-176B 训练期间用过的一个示例脚本 [slurm-status.py](slurm-status.py)。这个看门狗如果发现某个作业既不在运行也不在排队，就会发送一封邮件，同时还会把检查结果写入主训练日志文件。由于我们使用了[Crontab 模拟](../../orchestration/slurm/users.md#crontab-emulation)，只需要把 [slurm-status.slurm](slurm-status.slurm) 放进 `cron/cron.hourly/` 目录，之前启动的那个模拟 crontab 的 SLURM 调度器就会大约每小时启动一次这项检查。

这个 SLURM 作业的关键部分是：
```bash
tools/slurm-status.py --job-name $WATCH_SLURM_NAME 2>&1 | tee -a $MAIN_LOG_FILE
```
它告诉脚本要关注哪个作业名，你还能看到它把结果记录到一个日志文件中。

例如，如果你这样启动脚本：
```
tools/slurm-status.py --job-name my-training-10b
```
而当前状态报告显示：
```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
  JOBID    PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
    87     prod      my-training-10b  RUNNING 2-15:52:19 1-16:00:00  64    2023-10-07T01:26:28 node-[1-64]
```
那就一切正常。但如果 `my-training-10b` 作业没有出现在列表里，就会发出告警。

你只需修改路径和邮箱地址，就能以最小的改动把这些脚本适配到自己的需求。如果作业不是你启动的，就把 `whoami` 换成启动作业的那个用户名。`whoami` 只有在作业是你自己启动时才有效。


### 检查作业是否挂起的看门狗 {#is-job-hanging-watchdog}

如果应用在使用 `torch.distributed` 或类似机制，且挂起（hang）发生在某次集合通信期间，它最终会超时并抛出异常，从而重启训练，人们也可以发出作业被重启的告警。

然而，如果挂起发生在另一个可能没有超时的系统调用（syscall）期间，例如从磁盘读取，应用完全可能在那里挂起几个小时而无人察觉。

大多数应用都会定期记日志，例如大多数训练每隔几分钟就会记录最近 N 个步骤的统计信息。那么就可以检查日志文件是否在预期的时间范围内被更新过——如果没有——就发送告警。你可以自己写一个，或者使用 [io-watchdog](https://github.com/grondo/io-watchdog)（自 2019 年以来没有提交——应视为能用但不再维护，而不是活跃开发中）。



### 磁盘空间不足告警 {#low-disk-space-alerts}

下一个最大的问题是磁盘空间耗尽。如果你的检查点很大、保存频繁、又没有转储到别处，磁盘空间很容易被快速耗尽。而且，通常多名团队成员共用同一个集群，同事们可能一下子就消耗掉大量磁盘空间。理想情况下，你应当有一个只属于你的训练使用的存储分区，但这往往很难做到。无论哪种情况，你都需要在磁盘空间不足、需要腾出空间时及时知晓。

那么，告警应该在什么阈值触发？告警不能来得太早：比如用量到 50% 就开始发，用户很快就会对这些告警视而不见。但百分比也并非总是适用：如果你和别人共用一块巨大的磁盘，那块磁盘的 5% 可能相当于很多 TB 的剩余空间；而在一个小的分区上，哪怕还剩 25%，可能也只有几个 TB。所以真正重要的是，你要知道自己多久写一次检查点、每天需要多少 TB 的磁盘空间、以及实际有多少可用的磁盘空间。

使用案例：在 BLOOM 训练期间，我们每 3 小时写入一个 2.3TB 的检查点，因此每天要消耗 18.4TB！

此外，通常会有多个分区——IO 更快、专用于写检查点的分区，IO 较慢、用于存放代码和库的分区，还可能有各种其他正在使用的分区；只要训练的正常运行离不开它们的可用性，就必须把它们全部纳入监控。

这里还有一个坑——就分布式文件系统而言，并非所有文件系统都能可靠地提供你所申请磁盘空间的 100%。事实上，某些类型的文件系统最多只能可靠地使用已分配存储空间的大约 80%。问题在于，这类系统使用物理磁盘，并会在计划周期或触发事件时重新均衡数据，因此任何一块单独磁盘都可能达到 100% 容量而导致写入失败，进而使训练进程崩溃，哪怕 `df` 在分区上报告的用量只有 80%。训练 BLOOM-176B 时我们没有遇到这个问题，但训练 IDEFICS-80B 时遇到了——在那里的 80% 就是新的 100%。怎么知道自己有没有这个问题——通常你是在为训练做准备的过程中发现它的。

这还没完。还有 inode 可用性的问题，一些存储分区的 inode 配额并不大。Python 包出了名地包含成百上千个小文件，它们加起来占的总空间很小，但在一个虚拟环境里能累积出几万个文件，于是明明还有 TB 级的空闲磁盘空间，却耗尽了可用的 inode，然后发现训练崩溃了。

最后，许多分布式分区不会实时显示磁盘用量统计，更新可能要长达 30 分钟。

脚注：用 `df -ih` 查看 inode 配额和当前用量。

脚注：一些文件系统内部使用压缩，因此报告的磁盘用量可能比这些文件复制到别处后的实际占用要少，这会造成困惑。

下面是 BLOOM-176B 训练期间用过的 [fs-watchdog.py](./fs-watchdog.py)。这个看门狗只要有任何一项存储需求阈值没有满足，就会发送一封邮件，驱动它的是对应的 [fs-watchdog.slurm](./fs-watchdog.slurm)。

研究看门狗的代码你会看到，对每个分区我们同时监控磁盘用量和 inode。对某些分区，我们使用 HPC 提供的专用配额工具来获取即时统计，但这些工具并非对所有分区都可用，对那些分区只能退而使用 `df`，甚至是慢得多的 `du`。因此，把它适配到你自己的使用场景应该是很容易的。
### 应对缓慢的内存泄漏 {#dealing-with-slow-memory-leaks}

有些程序会产生极其微小的内存泄漏，这类问题非常难以调试。不要把这种泄漏与 MMAP 的用法混为一谈：在 MMAP 中，程序借用 CPU 内存来快速读取数据，内存占用看起来会随时间增长，但这并不是真正的泄漏，因为这些内存会在需要的时候被释放。你可以阅读 [A Deep Investigation into MMAP Not Leaking Memory](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/) 来了解其中的原因。

当然，最理想的情况是分析自己的软件并把泄漏修好，但有时泄漏来自第三方库，或者非常难以诊断，而且往往没有时间去查。

说到 GPU 显存，还存在内存碎片化这个可能的问题：随着时间推移，越来越多的细小块未使用内存段累积起来，使得 GPU 看起来仍有不少空闲显存，但当程序试图从这块内存中分配一个大张量时就会失败，报出类似这样的 OOM（Out Of Memory，内存耗尽）错误：

```
RuntimeError: CUDA out of memory. Tried to allocate 304.00 MiB (GPU 0; 8.00 GiB total capacity;
142.76 MiB already allocated; 6.32 GiB free; 158.00 MiB reserved in total by PyTorch)
```
在这个例子中，明明还有 6.32GiB 空闲，为什么 304MiB 反而分配不出来呢。

我的团队在训练 IDEFICS-80B 期间开发了一种办法。当时存在一处很小的 CPU 内存泄漏，往往过好几天就会把 CPU 内存耗尽。做法是在训练循环里装一个看门狗（watchdog），它会检查内存使用情况，一旦达到某个阈值就主动退出训练循环。下一个训练作业启动时，CPU 内存就全都被回收回来了。

脚注：机器学习训练的现实是，并非所有问题都能在有限资源下修好，而且很多时候一个扎实的变通方案能让人更快到达终点；相比之下，为了搞清问题出在哪里而“停下所有事情”、可能让训练推迟几周，反而更慢。例如我们训练 BLOOM-176B 时是带着 `CUDA_LAUNCH_BLOCKING=1` 跑的，因为不带它训练就会挂起；在多次尝试定位都失败之后，我们再也等不起了，只能照这样继续下去。幸运的是，这个通常用于调试目的、理论上会让某些 CUDA 操作变慢的环境变量，实际上并没有让我们的吞吐量有任何下降。但我们始终没搞清问题是什么，而今天我们完全不在乎了，因为我们已经转向其他项目，不受该问题影响。

这个思路和前面讨论过的终止开关与保存开关类似，只不过这里不是轮询某个文件是否出现，而是简单地观察常驻内存占用了多少。例如，下面是当操作系统显示虚拟 CPU 内存只剩 5% 时自动退出的写法：

```python
import psutil
for batch in iterator:
    total_used_percent = psutil.virtual_memory().percent
    if total_used_percent > 0.95:
        print(f"Exiting early since the cpu memory is almost full: ({total_used_percent}%)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

类似的启发式方法也可以用来为 GPU 显存用量设定阈值，只是需要留意 CUDA 张量缓存和 Python 垃圾回收的调度。因此要想拿到真实的显存用量，你得先运行垃圾回收器，再清空 CUDA 缓存，只有这样才能得到真实的显存统计，然后才能在 GPU 过于接近被占满时优雅地退出训练。

```python
import gc
import torch

for batch in iterator:
    gc.collect()
    torch.cuda.empty_cache()

    # 以 GiB 为单位获取显存用量，若剩余空闲 GPU 显存不足 2GiB 就退出
    free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info());
    if free < 2:
        print(f"Exiting early since the GPU memory is almost full: ({free}GB remain)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

脚注：除非万不得已，不要这样做，因为缓存能让事情更快。最好还是把碎片化的根源问题找出来并解决掉。例如，去查阅 [`PYTORCH_ALLOC_CONF`](https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables) 的文档中的 `max_split_size_mb`，因为它控制着内存如何被分配。一些框架，比如 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)，是通过在一开始就预分配张量、然后一次又一次复用来解决这个问题的，从而完全避免了碎片化。

脚注：这个简化的示例只适用于单节点。对于多节点，你需要从所有参与的节点收集统计信息，找出剩余内存最少的那个节点，并据此采取行动。


## 应对作业被强制抢占 {#dealing-with-forced-job-preemption}

前面你已经看到，训练可以用 [终止开关方案](#kill-switch) 优雅地停下来，这在需要按需停止或暂停训练时很有用。

在 HPC 集群上，SLURM 作业有一个最长运行时间限制。常见的上限是 20 小时。这是因为在 HPC 上资源在多个用户/多个课题组之间共享，因此每个人只分到一段做计算的时间片，随后作业会被强制停止，好让其他作业使用这些共享资源。

脚注：这也意味着，除非你的作业在集群上拥有最高优先级，否则你无法预先规划训练需要多久。如果你的优先级不是最高的，作业要等上几小时、有时甚至几天才能恢复运行，这并不罕见。

当然，你可以任由作业被杀掉，寄希望于自[上次保存检查点](#frequent-checkpoint-saving)以来没有跑太多迭代，然后让作业从该检查点恢复，但这非常浪费，最好避免。

高效的做法是在硬性时间上限到来、作业被 SLURM 杀掉之前优雅地退出。

首先，你需要弄清楚你的程序需要多少时间才能优雅收尾。这通常需要两个时长：

1. 如果你刚刚开始一次新的迭代，单次迭代跑完需要多久
2. 保存检查点需要多久

例如，如果一次迭代最多需要 2 分钟，保存检查点还需要 2 分钟，那么这段宽限时间你至少需要 4 分钟。为了保险起见，我会至少把它翻倍。稍微早一点退出并没有坏处，因为没有浪费任何资源。

所以，举例来说，假设你的 HPC 允许 100 小时的作业，那么你的 slurm 脚本里会写：
```
#SBATCH --time=100:00:00
```

### 方案 A：在启动时就告诉程序何时应开始退出流程 {#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process}
```bash
srun ... torchrun ... --exit-duration-in-mins 5990
```
100 小时是 6000 分钟，所以这里我们给程序留出 10 分钟来优雅退出。

程序启动时创建一个计时器，然后在每次新迭代开始前检查是否已达时间上限。如果达到了，就保存检查点并退出。

案例：你可以看到这在 [BLOOM 训练作业](https://github.com/bigscience-workshop/bigscience/blob/58d99c67f643d27b5765a73a2ee2d1ce0a4b2c6b/train/tr11-176B-ml/tr11-176B-ml.slurm#L97-L100) 中是如何设置的，以及随后在[这里](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998)是如何被执行的：

```python
        # 基于运行时长退出
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()
```

如你所见，由于训练是分布式的，我们必须把退出这一事件在所有 rank 之间同步。

你也可以把这个推导过程自动化：从正在运行的作业中取回 `EndTime`：
```bash
$ scontrol show -d job $SLURM_JOB_ID | grep Time
   RunTime=00:00:42 TimeLimit=00:11:00 TimeMin=N/A
   SubmitTime=2023-10-26T15:18:01 EligibleTime=2023-10-26T15:18:01
   AccrueTime=2023-10-26T15:18:01
   StartTime=2023-10-26T15:18:01 EndTime=2023-10-26T15:18:43 Deadline=N/A
```
然后在程序里把它与当前时间比较，据此设置优雅退出的提前量。从上面的输出可以看到，还能取到其他一些时间戳和时长。

### 方案 B.1：在结束前 X 分钟发送一个自定义信号 {#approach-b1-sending-a-custom-signal-x-minutes-before-the-end}

在你的 sbatch 脚本里，你可以这样设置：

```
#SBATCH --signal=USR1@600
```
这样 SLURM 就会在作业结束时间前 10 分钟向你的程序发送一个 `SIGUSR1` 信号。

脚注：通常 SLURM 调度器会在作业时间耗尽前约 30-60 秒发送 `SIGCONT`+`SIGTERM` 信号；而当时间正好耗尽、作业仍在运行时，它会发送 `SIGCONT`+`SIGTERM`+`SIGKILL` 信号。`SIGTERM` 可以被捕获并作出响应，但 30 秒不足以让一个大型模型训练程序优雅退出。

我们来演示一下信号的发送与捕获是如何工作的。在终端 A 里运行：
```bash
python -c "
import time, os, signal

def sighandler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```
它会打印出进程的 pid，例如 `4034989`，然后进入睡眠（模拟实际工作）。现在在终端 B 里用下面的命令向终端 A 中的 python 程序发送 `SIGUSR1` 信号：

```bash
kill -s USR1 4034989
```

程序会捕获这个信号，调用 `sighandler`，后者此时会打印自己被调用了，然后退出。

```
Signal handler called with signal 10
```
`10` 是 `SIGUSR1` 的数值。

下面是在 SLURM 配置下做同样的事情：

```bash
$ cat sigusr1.slurm
#SBATCH --job-name=sigusr1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:03:00
#SBATCH --partition=mypartition
#SBATCH --output=%x-%j.out
#SBATCH --signal=USR1@170

srun python -c "
import time, os, signal

def sighandler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```
在上面的 SLURM 脚本里，我们告诉 SLURM 在作业结束前 170 秒给它发一个信号，而作业本身被设置为运行 180 秒（3 分钟）。

当这个作业被调度之后：
```bash
sbatch sigusr1.slurm
```
作业启动 10 秒（`180-170`）后，它会退出，日志为：

```
58307
Signal handler called with signal 10
```

这说明该作业有一个 pid `58307`，它捕获了 `SIGUSR1`（即 `10`），然后退出了。

现在你明白了这套机制是如何运作的，那么就可以把立刻执行的 `exit(0)` 换成：设置一个“尽快退出”标志，跑完当前正在进行的这一次迭代，检查该标志是否已置位，保存检查点，然后退出。这与 [方案 A](#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process) 中展示的代码非常相似。


### 方案 B.2：选择把信号发送给哪个进程 {#approach-b2-choosing-which-process-to-send-the-signal-to}

那如果你的主程序并不是由 `srun` 启动的呢——如果你使用了一个中间启动器（launcher），比如 `torchrun` 或 `accelerate`，上面那段做法就行不通了，因为 `SIGUSR1` 很可能不会从启动器传递给它的子进程。在这种情况下，我们需要一个比前面展示的那个稍微复杂一些的 slurm 脚本。

我们必须把：
```
#SBATCH --signal=USR1@600
```
替换成：
```
#SBATCH --signal=B:USR1@600
```

多加的 `B:` 告诉 SLURM 不要把信号发给 `srun` 进程（启动器），而是发给 `sbatch` 所在的 shell。

接着我们还必须把 SLURM 脚本的结尾从典型的基于启动器的代码：

```bash
CMD="python -u -m torch.distributed.run ... train.py ..." # 这里换成真实的命令
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE

```
改成这样：
```bash
trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|deepspeed|torchrun|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; \
wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # 这里换成真实的命令
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

由于前面的 `--signal=B:USR1@600` 现在会把信号发给 `sbatch` 所在的 shell，我们就可以捕获它并对此做点什么，而 `trap` 那一行做的就是这件事。

传给 `trap` 的信号处理函数里那段巧妙的代码，会找出 `accelerate`、`deepspeed`、`torchrun` 或 `torch.distributed.run` 这类启动器的所有直接子进程，并把 `SIGUSR1` 信号发送给它们。

最后一点改动是：为了让 `trap` 能够生效，我们需要把 `srun` 放到后台运行——所以我们在 `srun` 命令的末尾加上了 `&`，并且必须加上 `wait`，这样 `sbatch` 所在的 shell 就不会在 `srun` 结束之前退出。

你那边捕获信号处理函数的 python 代码，与方案 B.1 中保持一样。

下面把 SLURM 脚本中重要的部分放在一起看：

```bash
$ cat launch.slurm
#!/bin/bash
[...]
#SBATCH --partition=dev
#SBATCH --signal=B:USR1 # 自定义抢占信号
[...]

trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|torchrun|deepspeed|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # 这里换成真实的命令
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

而你原本可能长这样的训练循环：
```bash
$ cat train.py

for batch in dl:
    train_iteration(batch)
```

现在它会变成：
```bash
$ cat train.py

import signal
import sys

pre_emption_activated = False
def activate_pre_emption(sig, frame):
    global pre_emption_activated
    print("SIGUSR1 received, saving checkpoint")
    pre_emption_activated = True

signal.signal(signal.SIGUSR1, activate_pre_emption)

for batch in dl:
    train_iteration(batch)

    if pre_emption_activated:
        save_checkpoint()
        sys.exit()
```

当然，在真实软件里你大概会在 trainer 对象中设置一个标志，而不是使用 `global`，但就这份简短的演示而言，这样已经足够了。

如果你想测试这个方案，只要把 SLURM 脚本的头部改成：

```
#SBATCH --time=0:05:00
#SBATCH --signal=B:USR1@60
```

这里我们告诉 SLURM 这个作业只运行 5 分钟（`--time=0:05:00`），并请它在 5 分钟到期前 `60` 秒向我们的 `sbatch` 脚本发送 `SIGUSR1`，也就是作业开始 4 分钟之后。



### 基于 QoS 的 SLURM 抢占 {#qos-based-slurm-preemption}

到目前为止我们还没有讨论使用服务质量（Quality of Service，QoS）时会发生什么，它同样可能强制抢占一个已经在运行的作业。这个功能与“作业的分配时间即将结束”那类抢占是一样的，区别在于它可能在任何时刻发生，而不是在作业结束前 X 秒。

考虑这样一个 SLURM 设置：你有 `--qos=high`，它可以抢占 `--qos=low` 的作业，而低优先级作业有 10 分钟的关闭宽限时间：

```bash
$ sacctmgr show qos format=name,priority,preempt,MaxTRESPerUser,GraceTime,Preempt,Flags
      Name   Priority     MaxTRESPU  GraceTime    Preempt                Flags
---------- ---------- ------------- ---------- ---------- --------------------
       low          0                 00:10:00
      high          0                 00:00:00        low
```

这与基于时间的抢占非常相似，只是这里的宽限时间是硬编码的，用户无法修改。

如果一个作业以 `--qos=high` 启动，而可用的节点不够多，SLURM 就会踢掉几个低优先级作业，把这些节点腾出来给高优先级作业用。

默认情况下 `GraceTime` 可能很短，不足以让你的程序在被抢占时安全收尾——这种情况下，请让你的系统管理员把它的时长提高到符合你需求的数值。

除此之外，方案 B.1 和 B.2 中介绍的同样那些方案，对这种强制抢占同样适用。
