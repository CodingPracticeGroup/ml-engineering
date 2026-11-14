# 面向用户的 SLURM {#slurm-for-users}

## 快速开始 {#quick-start}

只需复制这个 [example.slurm](./example.slurm)，并按你的需要进行调整。

## SLURM 分区 {#slurm-partitions}

在本文中，我们会使用一个示例配置，其中包含这 2 个集群名称：

- `dev`
- `prod`

要查明节点的主机名及其可用性，使用：

```bash
sinfo -p dev
sinfo -p prod
```

`slurm.conf` 的路径因站点而异（通常位于 `/etc/slurm/` 或 `/opt/slurm/etc/` 下）；要查看这个集群是从哪里加载它的：

```bash
scontrol show config | grep -i slurmconf
```

要查看所有分区的配置：

```bash
scontrol show partition
```

## 等待资源分配的时间 {#wait-time-for-resource-granting}

```bash
squeue -u `whoami` --start
```
它会显示任何挂起（pending）的作业预计何时开始运行。

如果其他人在预留结束之前就取消了自己的预留，你的作业也可能提前开始。



## 通过依赖关系申请资源分配 {#request-allocation-via-dependency}

想要在当前的某个已排队作业结束时（无论它仍在运行还是尚未开始）就排上新作业，可以使用依赖机制：告诉 `sbatch`，等当前正在运行的作业成功结束后再启动新作业，用法如下：

```bash
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID tr1-13B-round1.slurm
```

使用 `--dependency` 的等待时间可能比使用 `--begin` 更短，因为如果传给 `--begin` 的时间相对于最后一个作业停止的时间哪怕只留了几分钟的余量，调度器都可能已经先启动了一些其他作业，即使它们的优先级比我们的作业低。这是因为调度器在指定时间到来之前会忽略任何带 `--begin` 的作业。


## 在计划的时间进行资源分配 {#make-allocations-at-a-scheduled-time}

要把资源分配推迟到给定时间，使用：
```bash
salloc --begin=YYYY-MM-DDTHH:MM:SS
```

`sbatch` 同理。

它只是会在请求的时间点把作业放入队列，就像你恰好在的那个时间执行了这条命令一样。如果那时资源可用，分配会立即给出；否则会排队等待。

有时使用相对的开始时间很有用，而且可以使用其他格式。示例：

```
--begin now+2hours
--begin=16:00
--begin=now+1hour
--begin=now+60  # 默认单位是秒
--begin=2010-01-20T12:34:00
```

时间单位可以是 `seconds`（默认）、`minutes`、`hours`、`days` 或 `weeks`。

## 无 60 分钟时间限制的预分配节点 {#preallocated-node-without-time-60min-limit}

这对反复进行的交互式实验非常有用——这样就不必每次为了推进工作而等待资源分配。因此策略是：一次性为较长的时间段申请资源，然后利用这份分配运行交互式的 `srun` 作业。

把 `--time` 设置为你想要的时间窗口（例如 6h）：
```bash
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time=6:00:00 bash
salloc: Pending job allocation 1732778
salloc: job 1732778 queued and waiting for resources
salloc: job 1732778 has been allocated resources
salloc: Granted job allocation 1732778
```
接下来，通过传入 `salloc` 的作业 id，用这台预留的节点多次运行作业：
```bash
srun --jobid $SLURM_JOBID --pty bash
```
如果是在 `salloc` 启动的 `bash` 里面运行，可以直接这样执行。但也可以从另一个 shell 启动，只是那时需要显式设置 `--jobid`。

如果这个 `srun` 作业超时或被手动退出，你可以在这同一台预留节点上重新启动它。

当然，`srun` 可以直接调用真正的训练命令，而不仅仅是 `bash`。

重要：分配单个节点时，分配到的 shell 并不在节点上（从来都不是）。你必须先弄清楚该节点的主机名（分配获批时会报告，也可以通过 `squeue` 查看），然后 `ssh` 登录到它。

结束后，要释放资源，退出 `salloc` 启动的 shell，或者执行 `scancel JOBID`。

这台预留节点在整个分配期间都会被计入小时用量，所以用完请尽快释放。

其实，如果只是一台节点，更省事的做法是一开始就不使用 `salloc`，而直接使用 `srun`，它既会完成分配，又会给你一个可用的 shell：
```bash
srun --pty --partition=dev --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --time=60 bash
```

## 超线程 {#hyper-threads}

默认情况下，如果 CPU 启用了[超线程](https://en.wikipedia.org/wiki/Hyper-threading)（Hyper-Threads，HT），SLURM 就会使用它。如果你不想使用 HT，必须指定 `--hint=nomultithread`。

脚注：HT 是 Intel 专有的叫法，通用概念是同时多线程（simultaneous multithreading，SMT）

例如，对于一个每节点 2 个 CPU、每个 CPU 24 个核心、每核心 2 个超线程的集群，节点上一共有 96 个超线程或 48 个 CPU 物理核心可用。因此，要充分利用这台节点，你需要配置下面二者之一：

```
#SBATCH --cpus-per-task=96
```
或者，如果你不想用 HT：
```
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
```

最后这种方式会为每个核心分配一个线程，在这种模式下只有 48 个 CPU 核心可用。

注意，取决于你的应用，这 2 种模式之间可能存在相当大的性能差异。因此请两种都试一下，看看哪种效果更好。

在 AWS 等某些环境上，使用 `--hint=nomultithread` 时 all-reduce 吞吐量会急剧下降！而在某些其他环境上情况正好相反——不开 HT 吞吐量反而更差！

要检查你的实例是否启用了 HT，运行：

```bash
$ lscpu | grep Thread
Thread(s) per core: 2
```

如果是 `2` 就表示启用了 HT，如果是 `1` 则表示没有启用。


## 复用资源分配 {#reuse-allocation}

例如，当你想在完全相同的节点分配上运行各种作业时。

在一个 shell 中：
```bash
salloc --partition=prod --nodes=16 --ntasks=16 --cpus-per-task=96 --gres=gpu:8 --time=3:00:00 bash
echo $SLURM_JOBID
```

在另一个 shell 中：
```bash
export SLURM_JOBID=<JOB ID FROM ABOVE>
srun --jobid=$SLURM_JOBID ...
```

在节点上运行某些诊断作业时，你可能需要设置 `--gres=gpu:0`。例如，我们来检查所有主机的共享内存：
```bash
srun --jobid 631078 --gres=gpu:0 bash -c 'echo $(hostname) $(df -h | grep shm)'
```


## 选择特定节点 {#specific-nodes-selection}

要排除特定节点（当你明知某些节点坏了、但状态仍是 IDLE 时很有用）：

```bash
sbatch --exclude nodeA,nodeB
```
或者通过：`#SBATCH --exclude ...`

要使用特定节点：

```bash
sbatch --nodelist=nodeA,nodeB
```
也可以用短选项 `-w` 代替 `--nodelist`


管理员还可以在 `slurm.conf` 中定义一个 `feature=example`，这样用户就可以通过 `--constraint=example` 请求这组节点。


## 通知正在运行的作业结束 {#signal-the-running-jobs-to-finish}

由于每个 SLURM 作业的运行时间有限，可以这样配置：在分配的时间结束前一段期望的时长，向程序发送一个自选的信号。
```
--signal=[[R][B]:]<sig_num>[@<sig_time>]
```
至于实际可用的模式——在墙钟时限前几分钟捕获 `SIGUSR1`（或类似的信号），完成当前步骤，保存检查点，然后不再开始新一轮就退出——参见故障容错章节中的[在结束前 X 分钟发送自定义信号](../../training/fault-tolerance/README.md#approach-b1-sending-a-custom-signal-x-minutes-before-the-end)。



## 作业详细信息 {#detailed-job-info}

虽然大多数有用的信息都存在于各种 `SLURM_*` 环境变量中，但有时仍会缺少某些信息。遇到这种情况时使用：
```bash
scontrol show -d job $SLURM_JOB_ID
```
然后解析出需要的内容。


对于已经运行结束的作业，使用：
```bash
sacct -j JOBID
```

这条命令还有助于发现该分配上是否已有 `srun` 作业在运行（包括已结束或已取消的那些）。例如，你可能想通过 `scancel <jobid>.<step-id>` 杀掉某个失控的 `srun` 步骤（step），而上述命令正是用来找到那个 `<step-id>` 的。即使你取消了全部步骤作业，只要主作业是交互式作业，它仍会继续运行。

要查看更多细节：
```bash
sacct --format=jobid,start,end,state,exitcode,nodelist%300 -j JOBID
sacct -j JOBID --long
```

或者，把所有作业连同其子步骤一起列出，但只限定某个特定分区、且只针对你自己的用户：

```bash
sacct -u `whoami` --partition=dev  --format=jobid,start,end,state,exitcode,nodelist%300
sacct -u `whoami` --partition=prod --format=jobid,start,end,state,exitcode,nodelist%300
```

要查看某个作业是如何启动的，以及它所有 `srun` 子步骤的命令行：
```bash
sacct -j JOBID -o submitline -P
```

## 显示作业 {#show-jobs}


只显示我自己的作业：
```bash
squeue -u `whoami`
```

按作业 id 显示作业：
```bash
squeue -j JOBID
```

显示某个特定分区的作业：
```bash
squeue --partition=dev
```

## 获取作业的信息 {#getting-information-about-the-job}

在 slurm 文件内部，可以获取当前作业分配情况的信息。

获取已分配节点的主机名，并基于此做一些有用的派生信息：
```bash
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export NUM_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
```

## 环境变量 {#environment-variables}

`#SBATCH --export=...` 控制把**提交 shell** 中的哪些变量复制到作业里。在标准版 Slurm 上默认是 `ALL`（有些站点会通过 `SBATCH_EXPORT` / `SLURM_EXPORT_ENV` 或 cli_filter 覆盖该默认值）。无论哪种模式，`SLURM_*` 变量始终存在。你在作业脚本内部 `export KEY=value` 的任何变量，也总是能被启动的程序看到——这与 `--export` 无关，后者只决定从提交 shell 继承哪些内容。4 种有用的形式：

1. **完整的提交 shell 环境（默认）：**

```
#SBATCH --export=ALL
```
在标准版 Slurm 上，这与省略 `--export` 相同——单独写这一行在那里没有任何作用。只有在需要撤销站点的默认值 `NONE` 时，或像（4）那样与附加项组合使用时才需要它。

2. **最小环境：**

```
#SBATCH --export=NONE
```
丢弃提交 shell 的环境。当你想要一个可复现、不继承登录 shell 中碰巧导出过的任何变量的作业时，这很有用。

3. **只要这些变量（替换列表）：**

```
#SBATCH --export=KEY1,KEY2=value
```
**只**传播列出的变量（这里是：`KEY1` 的当前值，以及设为 `value` 的 `KEY2`）。提交 shell 中的其他内容全部丢弃——这不是在 `ALL` 基础上的追加。

4. **完整环境并设置/覆盖：**

```
#SBATCH --export=ALL,KEY=value
```
保留完整的提交 shell 环境，**并且**设置或覆盖 `KEY`。如果你既想在提交时拥有"全部"环境，又要额外加一个变量，`ALL` 必须出现在列表中——否则就变成了情况（3）。

## 把紧凑节点列表转换为展开节点列表 {#convert-compact-node-list-to-expanded-node-list}

有时 SLURM 工具会给你一个这样的字符串：`node-[42,49-51]`，要把它展开成 `node-42,node-49,node-50,node-51` 还需要写些代码，不过有一个专门的工具可以处理这个：

```bash
$ scontrol show hostnames node-[42,49-51]
node-42
node-49
node-50
node-51
```
搞定！

案例：例如，你可能想获取因作业退出太慢而被排空（drain）的节点列表，而其实节点本身并没有真正的问题。那么下面这个一行命令就会以展开格式给出这类节点的列表，之后你可以写个脚本遍历这个列表，也许先确认相关进程此时已经消亡，再把它们重新设为可用：
```bash
sinfo -R | grep "Kill task failed" | perl -lne '/(node-.*[\d\]]+)/ && print $1' | xargs -n1 scontrol show hostnames
```

## 把 SLURM_JOB_NODELIST 转换为 hostfile {#convert-slurm_job_nodelist-into-a-hostfile}

有些多节点启动器需要一份 `hostfile`——下面是生成它的方法。展开使用 `scontrol show hostnames`，与上文[把紧凑节点列表转换为展开节点列表](#convert-compact-node-list-to-expanded-node-list)是同一个工具，所以 `node-[42,49-51]` 这样的混合形式也可以：

```bash
# 用法：
# makehostfile > hostfile
# 依赖 SLURM_STEP_GPUS=0,1,2... 来获取每个节点的 gpu 槽位数
function makehostfile() {
  local slots
  slots=$(perl -F, -lane 'print scalar @F' <<<"${SLURM_STEP_GPUS:-0}")
  scontrol show hostnames "$SLURM_JOB_NODELIST" | perl -slne 'print "$_ slots=$s"' -- -s="$slots"
}
```

## 别名 {#aliases}

一些好用的别名

```bash
alias myjobs='squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"'
alias groupjobs='squeue -u foo,bar,tar -o "%.16i %u %9P %26j %.8T %.10M %.8l %.6D %.20S %R"'
alias myjobs-pending="squeue -u `whoami` --start"
alias idle-nodes="sinfo -p prod -o '%A'"
```




## 僵尸进程 {#zombies}

如果各节点上还有遗留的僵尸进程，发一条命令把它们全部杀掉。

```bash
srun pkill python
```

## 深入查看 SLURM 记账 {#detailed-access-to-slurm-accounting}

`sacct` 显示 SLURM 作业记账日志或 SLURM 数据库中所有作业和作业步骤的记账数据。

所以这是一个分析历史事件的绝佳工具。

例如，要查看最近的 gpu 作业是用哪些节点运行的：

```bash
sacct -u `whoami` --partition=dev --format=jobid,start,end,state,exitcode,nodelist%300
```

这里的 `%300` 告诉它输出使用 300 个字符的宽度，以免内容被截断。

更多字段和信息字段请见 `man sacct`。



## 队列 {#queue}


### 取消作业 {#cancel-job}

要取消一个作业：
```bash
scancel [jobid]
```

要取消你的全部作业：
```bash
scancel -u <userid>
```

要取消你在某个特定分区上的全部作业：
```bash
scancel -u <userid> -p <partition>
```

### 提示 {#tips}

- 如果你看到 `salloc` 的交互式作业被安排到远晚于你需要的时刻才运行，试着取消这个作业，改申请更短的时段——往往更近的时间窗口有可能给更短的时间分配留出空位。


## 日志 {#logging}

如果我们需要按节点把日志拆分到不同的日志文件，加入 `%N`（用于短主机名），写成：

```
#SBATCH --output=%x-%j-%N.out
```

这样我们就能判断某个特定节点是否表现异常——例如有一块坏掉的 GPU。当入口点用 `@record` 包装时，`torchrun` 能把首次失败归因到某台主机/rank，但未经装饰的脚本仍然只会得到一个裸的堆栈跟踪，而单份汇总并不能取代每个节点一个日志文件。


## 查看节点状态 {#show-the-state-of-nodes}
```bash
sinfo -p PARTITION
```

非常有用的命令是：
```bash
sinfo -s
```

然后查看主要统计信息，例如：

```
NODES(A/I/O/T) "allocated/idle/other/total".
597/0/15/612
```
可见 612 台节点中有 597 台已被分配，0 台空闲，另有 15 台由于其他各种原因不可用。

要查看某个特定分区：

```bash
sinfo -p gpu_p1 -o "%A"
```

得到：
```
NODES(A/I)
236/24
```

这样你就能看到 4x v100-32g 分区（`gpu_p1`）上是否有节点可用。


### sinfo 状态 {#sinfo-states}


- idle：没有作业在运行
- alloc：节点已分配给当前正在执行的作业
- mix：节点上部分 CPU 已被分配，其余空闲
- drain：由于管理原因，该节点不可用
- drng：该节点正在运行作业，但作业完成后将因管理原因不可用


### 节点状态代码 {#node-state-codes}

节点状态后面可能跟着一个具有特殊含义的单字符。它是以下之一：

- `*`：该节点当前没有响应，不会被分配任何新工作。如果节点持续无响应，它将被置于 DOWN 状态（COMPLETING、DRAINED、DRAINING、FAIL、FAILING 状态的节点除外）。
- `~`：该节点当前处于断电关闭状态。
- `#`：该节点正在加电或配置中。
- `!`：该节点正等待断电。
- `%`：该节点正在断电中。
- `$`：该节点当前处于带有 "maintenance" 标记的预留中。
- `@`：该节点正等待重启。
- `^`：已对该节点发出重启指令。
- `-`：回填调度器已为更高优先级的作业预订了该节点。

### 作业状态代码 {#job-state-codes}

- `CD` | Completed：该作业已成功完成。
- `CG` | Completing：该作业正在结束，但仍有部分进程在活动。
- `F` | Failed：该作业以非零退出码终止，未能执行成功。
- `PD` | Pending：该作业正在等待资源分配。它最终会运行。
- `PR` | Preempted：该作业因被另一个作业抢占而终止。
- `R` | Running：该作业当前已分配到节点并正在运行。
- `S` | Suspended：正在运行的作业被停止，其核心已释放给其他作业。
- `ST` | Stopped：正在运行的作业被停止，但其核心仍被保留。


### 被排空的节点 {#drained-nodes}

要查看所有被排空（drained）的节点以及排空的原因（修改 `%50E` 可以让原因字段更长/更短）
```
% sinfo -R -o "%50E %12U %19H %6t %N"
```

或者，如果你想要简短的输出，只使用 `-R`：

```
% sinfo -R
```



## 作业数组 {#job-arrays}

要运行一系列作业，使当前作业一结束就立刻调度下一个 slurm 作业，我们使用作业数组。

先从只有 10 个这样的作业开始：

```bash
sbatch --array=1-10%1 array-test.slurm
```

`%1` 把这个作业数组中同时运行的任务数限制为 1。不加它的话，它会尝试一次性运行所有作业，有时我们可能正想要这样（那时去掉 %1 即可），但在训练时我们需要一次只运行一个作业。

或者，像往常一样，这个参数也可以作为脚本的一部分：
```
#SBATCH --array=1-10%1
```

下面是一个玩具 slurm 脚本，可以用来看看它是如何工作的：

```bash
#!/bin/bash
#SBATCH --job-name=array-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 关键——每个节点每个分布式进程只能有 1 个任务！
#SBATCH --cpus-per-task=1            # 每个任务的核数
#SBATCH --time 00:02:00              # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --error=%x-%j.out            # 错误文件名（与输出同名，这样只需查看一个文件）
#SBATCH --partition=dev

echo $SLURM_JOB_ID
echo "I am job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
date
sleep 10
date
```

注意 `$SLURM_ARRAY_JOB_ID` 与 `$SLURM_JOB_ID` 相同，而 `$SLURM_ARRAY_TASK_ID` 是作业的索引。

要查看正在运行的作业：
```bash
$ squeue -u `whoami` -o "%.10i %9P %26j %.8T %.10M %.6D %.20S %R"
     JOBID PARTITION                       NAME    STATE       TIME  NODES           START_TIME NODELIST(REASON)
591970_[2-   dev             array-test  PENDING       0:00      1  2021-07-28T20:01:06 (JobArrayTaskLimit)
```
现在正在运行的是 2 号作业。

要取消整个数组，像平常一样取消作业 id（`_` 之前的数字）即可：
```bash
scancel 591970
```

要取消某个特定的作业：
```bash
scancel 591970_2
```

如果希望日志文件名中包含数组 id，加上 `%A_%a`：

```
#SBATCH --output=%x-%j.%A_%a.log
```

更多细节见 https://slurm.schedmd.com/job_array.html

## 作业数组列车及其挂起与恢复 {#job-array-trains-and-their-suspend-and-release}

在这个做法中，我们要达成 2 个目标：

1. 允许修改下一个作业的 slurm 脚本
2. 允许挂起和恢复作业数组，在还没准备好继续运行作业时也不丢失队列中的排位

SLURM 是一个非常严苛的环境，一个小错误就可能让你付出几天的等待时间。但有一些策略可以减轻这种严苛。

SLURM 作业在队列中有"资历（age）"的概念，除了项目优先级之外，它还决定一个作业何时被调度运行。如果你刚提交一个新作业，它没有任何"资历"，与更早进入队列的作业相比，通常会被安排最后运行。当然，如果这个新作业来自高优先级项目，它就会推进得更快。

那么，当你需要修改正在运行的脚本中的某些东西、或者例如要切换到另一个脚本时，如何保住"资历"而不失去它？方法如下。

思路是这样的：

1. 用 `sbatch` 提交一个长的作业数组，例如 `-array=1-50%1`
2. 在 slurm 脚本里除了 `source another-script.slurm` 之外不放任何代码——这样你就可以在下一个作业开始前修改目标脚本，或把它符号链接（symlink）到另一个脚本
3. 如果你需要停下作业数组这列"列车"——不要取消它，而是让它处于挂起（hold）状态，这样不会丢失你在队列中的排位
4. 准备好继续时——释放（release）这个作业数组——只有在挂起状态期间的那段时间不计入资历，之前的全部资历都会被保留。

正在运行的作业的节点数、时间、硬件和分区是不能修改的，但你可以通过 `scontrol update jobid=<desired_job_id> numnodes=<new number> partition=<new partition>` 来修改作业数组中处于挂起状态的作业。

如果你有 `sudo` 权限，那么也可以修改当前作业的时长。

下面是一个例子：

创建一个作业脚本：

```bash
$ cat train-64n.slurm
#!/bin/bash
#SBATCH --job-name=tr8-104B
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # 关键——每个节点每个分布式进程只能有 1 个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核数
#SBATCH --gres=gpu:8                 # GPU 数量
#SBATCH --time 20:00:00              # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=dev

source tr8-104B-64.slurm
```
像这样启动它：
```bash
sbatch --array=1-50%1 train-64.slurm
```

现在，你可以在下一个作业运行前轻松编辑 `tr8-104B-64.slurm`，然后既可以让当前作业自然跑完（如果你希望如此），也可以在需要中止它时直接杀掉当前正在运行的作业，例如 `1557903_5`（而不是作业数组 `1557903`），让列车接着上次停下的地方继续，只是使用编辑后的脚本。

好处在于，这不需要修改原始脚本（本例中是 `tr8-104B-64.slurm`），而且后者仍然可以单独启动。

那么，如果出了问题、你需要 10 分钟或 10 个小时来修复，该怎么办？这种情况下，我们用下面的命令让列车挂起：

```bash
scontrol hold <jobid>
```

其中 <jobid> 可以是"普通"作业、作业数组的 id，或作业数组某个步骤的 id

然后，准备继续时释放该作业：

```bash
scontrol release <jobid>
```


## 退出 shell 时让 salloc 分配保持存活 {#how-to-keep-an-salloc-allocation-alive-while-exiting-its-shell}

如果你像这样分配了一台节点：

```bash
salloc --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00 bash
```
而你退出了 shell，或者你的 ssh 连接断开了，这份分配就会丢失。

如果你想建立一个在退出 shell 后仍能存活的分配，使用 `--no-shell`，并且不要带 `bash`，像这样：

```bash
salloc --no-shell --partition=dev --nodes=1 --ntasks-per-node=1 --time=1:00:00
```
现在如果你需要加入这个会话，见[如何以交互方式重新接入已分配的节点](#how-to-rejoin-the-allocated-node-interactively)。

但要注意，如果你 `ssh` 到已分配的节点，以常规方式启动了某个东西，然后又关闭连接，那个作业就会丢失，因为发起连接的 shell 会向它的子进程发送 `SIGHUP`。要避免这一点并让作业继续运行，请使用 `nohup`，同时把程序放到后台进程中。示例：

```
nohup my-program &
```

`nohup` 会忽略 `SIGHUP`，会把 stderr 重定向到 stdout，并把 stdout 追加写入一个特殊文件 `nohup.out`。如果你想控制标准流写到什么位置，就使用普通的 stdout 重定向 `>` 或 `>>`，例如：
```
nohup my-program >> some-file.txt &
```

如前所述，程序也是通过 `&` 被放到后台的。

现在你可以放心断开连接，等你回来时程序仍在运行。

这个方案能防止程序退出，但由于标准流已被重定向，你再次连接时将无法与它正常交互。当然，你仍然可以通过 pid 杀掉该程序、改变它的 `nice` 状态等，就像对其他任何进程做的那样。

但如果你希望断开后重连还能继续使用程序，就得使用[终端复用器](https://en.wikipedia.org/wiki/Terminal_multiplexer)程序，例如 [`tmux`](https://github.com/tmux/tmux) 或 GNU [`screen`](https://www.gnu.org/software/screen/)，它们在节点上运行一个守护进程（daemon），让你重新连接时重新获得对程序的正常控制权。此外还有 [`mosh`](https://github.com/mobile-shell/mosh) 和其他类似工具，可以进一步帮助这一过程。



## 如何以交互方式重新接入已分配的节点 {#how-to-rejoin-the-allocated-node-interactively}

要想在同一个作业中拥有多个交互式 shell，应当使用 `--overlap`。

例如，在控制台 A 中，我们先分配一台单节点：
```bash
$ salloc --partition=dev --nodes=1 --ntasks-per-node=1 --cpus-per-task=26 --gres=gpu:1 --time=2:00:00 bash
salloc: Granted job allocation 1916
salloc: Nodes my-node-1 are ready for job
```

在控制台 B 中：
```bash
$ srun --overlap --pty --jobid 1916 bash
```
上面这一步可以在任意多个控制台中重复。

如果是第一个伪终端 shell，你甚至不需要 `--overlap`，但对额外的 shell 就需要它了。

如果你最初是通过 `srun --pty` 分配的节点，做法相同：
```bash
srun --pty -p dev --gpus 8 --time=2:00:00 bash
```

当然，你也可以通过 `ssh` 访问该节点，但如果你的 SLURM 被配置成了各种虚拟化方式（例如每个用户只分配少量 GPU，或把 `/tmp/` 或 `/scratch` 虚拟化并在退出时自动清理），那么从 `ssh` 看到的景象就不一样。例如，如果一个作业分配了 2 块 GPU，ssh 的 shell 会显示出全部 GPU 而不只是那 2 块——所以如果你和别人共享这台节点，这样就不太合适。

这对多节点分配同样有效，默认你会得到该分配中第一台节点上的交互式 shell。如果你想进入某台特定节点，用 `-w` 指定它。例如，假设你分配到了 `node-[1-4]`，而你想进入 `node-3`，那就指定：
```bash
srun --pty -p dev --gpus 8 --time=2:00:00 -w node-3 bash
```
如果它报错：
```bash
srun: error: Unable to create step for job 1930: Invalid generic resource (gres) specification
```
把 `--gres=gpu:8` 设置加回来即可。如果你最初的分配命令已经使用了这个标志，就不必这样做。




## 问题排查 {#troubleshooting}



### `SLURM_PROCID` 的提前插值 {#slurm_procid-early-interpolation}

在多节点配置下使用 SLURM 时，把这一点设置正确至关重要：
```
"--machine_rank \$SLURM_PROCID"
```
它不能被提前插值，因为如果写成 `"--machine_rank $SLURM_PROCID"`，启动器就会挂起（hang）。

最好把启动器与程序隔离开，像这样：

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3333
ACCELERATE_CONFIG_FILE=path/to/accelerate.config.yaml # 修改我
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    "
PROGRAM="myprogram.py"

CMD="$LAUNCHER $PROGRAM"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --unbuffered \
    --jobid $SLURM_JOBID \
    "

srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a main_log.txt
```

这样启动器就总能正常工作，用户只需要调整 `PROGRAM` 变量。

使用 `torchrun` 时：

```bash
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3333
# 注意 `\$SLURM_PROCID` 和 `\$(hostname ...)`——在 `srun` 时才插值，而不是在这里
LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role \$(hostname -s|tr -dc '0-9'): \
    --tee 3 \
    "
```

完整可用的示例见[使用 SLURM 的单节点与多节点启动器](launchers/README.md)。


### 节点数不匹配 {#mismatching-nodes-number}

如果 pytorch 启动器失败，往往意味着 SLURM 的节点数和启动器的节点数不匹配，例如：

```bash
grep -ir nodes= tr123-test.slurm
#SBATCH --nodes=40
NNODES=64
```

这样是不行的，二者必须匹配。

你可以在脚本里加一个合理性检查：

```bash
#!/bin/bash
#SBATCH --job-name=test-mismatch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 关键——每个节点每个分布式进程只能有 1 个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核数
#SBATCH --gres=gpu:8                 # GPU 数量
#SBATCH --time 0:05:00               # 最大执行时间（HH:MM:SS）
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

[...]

NNODES=2

# 合理性检查：确保 NNODES 与 `#SBATCH --nodes` 一致，前提是你使用了 NNODES 变量
if [ "$NNODES" != "$SLURM_NNODES" ]; then
    echo "Misconfigured script: NNODES=$NNODES != SLURM_NNODES=$SLURM_NNODES"
    exit 1
fi

[...]
```

或者你也可以直接这样：

```bash
#SBATCH --nodes=2
[...]
NNODES=$SLURM_NNODES
```

这样它就永远是正确的
### 找到有故障的节点并把它们排除 {#find-faulty-nodes-and-exclude-them}

有时候某个节点是坏的，这会导致无法进行训练，尤其是因为重启作业往往会再次分配到同一批节点。所以需要能够把坏节点隔离出来，并在 `sbatch` 中把它们排除掉。

要找到有故障的节点，写一个小脚本来报告所需检查项的状态即可。

例如，测试 cuda 是否在所有节点上都可用：
```bash
python -c 'import torch, socket; print(f"{socket.gethostname()}: {torch.cuda.is_available()}")'
```

以及只报告那些失败的节点：
```bash
python -c 'import torch, socket; torch.cuda.is_available() or print(f"Broken node: {socket.gethostname()}") '
```

当然，问题可能是别的——例如 GPU 无法分配显存，那就把测试脚本改成在 cuda 上做一次小规模分配。下面给出一种做法：

```bash
python -c "import torch; torch.ones(1000,1000).cuda()"
```

但由于我们需要在所有节点上运行这个测试脚本，而不只是在第一个节点上，slurm 脚本需要通过 `srun` 来运行它。因此我们的第一个诊断脚本可以写成：

```bash
srun --jobid $SLURM_JOBID bash -c 'python -c "import torch, socket; print(socket.gethostname(), torch.cuda.is_available())"'
```

因为存在引号方面的问题，我对它做了些微改动。

你随时可以把这个单行命令改写成一个真正的脚本，那样就没有引号问题了。

```bash
$ cat << EOT >> test-nodes.py
#!/usr/bin/env python
import torch, socket
print(socket.gethostname(), torch.cuda.is_available())
EOT
$ chmod a+x ./test-nodes.py
```

现在来创建一个驱动用的 slurm 脚本。这个测试只用几分钟的时间限额，这样 SLURM 会更快地让出资源：
```bash
#!/bin/bash
#SBATCH --job-name=test-nodes
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # 关键——每个节点每个分布式只运行 1 个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核数
#SBATCH --gres=gpu:8                 # GPU 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

source $six_ALL_CCFRWORK/start-prod   # 请修改——特定站点的环境设置
srun --jobid $SLURM_JOBID ./test-nodes.py
```
运行之后检查日志，看看是否有节点报告了 `False`，那些就是你要排除的节点。

现在既然找到了有故障的节点，就把它交给 `sbatch`：
```bash
sbatch --exclude=hostname1,hostname2 ...
```
`sbatch` 就会在资源分配时排除这些坏节点。

下面再介绍几种情况，以及在那些情况下如何找到坏节点：

### NCCL 故障 {#broken-nccl}

如果你要测试的东西需要分布式环境，事情就会更复杂一些。这里有一个测试 NCCL 能否工作的 slurm 脚本。它会设置好 NCCL 并检查 barrier（同步屏障）是否可用：

```bash
#!/bin/bash
#SBATCH --job-name=test-nodes-nccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 关键——每个节点每个分布式只运行 1 个任务！
#SBATCH --cpus-per-task=96           # 每个任务的核数
#SBATCH --gres=gpu:8                 # GPU 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=prod

source $six_ALL_CCFRWORK/start-prod   # 请修改——特定站点的环境设置

NNODES=2

GPUS_PER_NODE=4
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    "

export SCRIPT=test-nodes-nccl.py

cat << EOT > $SCRIPT
#!/usr/bin/env python
import torch.distributed as dist
import torch
import socket
import os
import fcntl

def printflock(*msgs):
    """ 打印 """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
header = f"{socket.gethostname()}-{local_rank}"
try:
    dist.barrier()
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} is OK")
except:
    printflock(f"{header}: NCCL {torch.cuda.nccl.version()} is broken")
    raise
EOT

echo $LAUNCHER --node_rank $SLURM_PROCID $SCRIPT

srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $SCRIPT'
```
这个脚本使用了 `printflock` 来解决多个进程打印输出交错的问题。


### GPU 显存检查 {#gpu-memory-check}

这个测试检查所分配节点上的每块 GPU 能否成功分配 77GiB 显存（例如用于测试 80GB 的 A100）（必须扣掉几 GiB 留给 cuda 内核）。


```python
import torch, os
import time
import socket
hostname = socket.gethostname()

local_rank = int(os.environ["LOCAL_RANK"]);

gbs = 77
try:
    torch.ones((gbs*2**28)).cuda(local_rank).contiguous() # 先在 cpu 上分配，然后移到 gpu
    print(f"{local_rank} {hostname} is OK")
except:
    print(f"{local_rank} {hostname} failed to allocate {gbs}GiB of accelerator memory")
    pass

time.sleep(5)


```


### 网络故障 {#broken-network}

节点的另一种问题是它的网络坏了，其他节点无法与它连接。

你很可能会遇到类似这样的错误：
```
work = default_pg.barrier(opts=opts)
RuntimeError: NCCL error in: /opt/conda/conda-bld/pytorch_1616554793803/work/torch/lib/c10d/ProcessGroupNCCL.cpp:825, unhandled system error, NCCL version 2.7.8
ncclSystemError: System call (socket, malloc, munmap, etc) failed.
```
下面是调试这个问题的方法：

1. 添加：
```bash
export NCCL_DEBUG=INFO
```
在 `srun` 命令之前加上它，然后重新运行你的 slurm 脚本。

2. 现在仔细研究日志。如果你发现：
```
r11i6n2:486514:486651 [1] include/socket.h:403 NCCL WARN Connect to 10.148.3.247<56821> failed : Connection refused
```
来看看是哪个节点拒绝接受连接。我们从上面的错误信息中拿到 IP 地址，并把它反向解析为名字：
```
nslookup 10.148.3.247
247.3.148.10.in-addr.arpa       name = r10i6n5.ib0.xa.idris.fr.
```

在 `sbatch` 命令中加入 `--exclude=r10i6n5`，并向集群管理员报告这个有故障的节点。


### 在所有节点上运行 py-spy 或其他监控程序 {#run-py-spy-or-any-other-monitor-program-across-all-nodes}

在处理进程挂起的问题时，下面介绍如何为每个进程自动记录 `py-spy` 跟踪信息。

当然，同样的过程也可以用来在某个作业的所有节点上运行某条命令。也就是说，它可以用于在正常运行期间执行某些操作——例如通过 `nvidia-smi` 转储每个进程的显存使用情况，或者运行其他任何需要运行的程序。



```bash
cd ~/prod/code/tr8b-104B/bigscience/train/tr11-200B-ml/

salloc --partition=prod --nodes=40 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:8 --time 20:00:00

bash 200B-n40-bf16-mono.slurm
```

在另一个 shell 中获取上面这个 `salloc` 的 JOBID：
```bash
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```
按上面的结果调整 jobid 和节点数量：
```bash
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```
不过 `--nodes=40` 多半是多余的，因为 `salloc` 已经知道节点数量了。

现在所有 `py-spy` 跟踪信息都会写入 `cwd` 下的 `trace-$nodename.out` 文件中。

关键在于要使用 `--gres=gpu:0`，否则第二个 `srun` 会一直阻塞，等待第一个 `srun` 释放 GPU。

另外，这里的假设是：某个装有 `py-spy` 的 conda 环境已经在 `~/.bashrc` 中被激活了。如果你的环境还没有这样做，就在上面的命令中、`py-spy` 命令之前，加上加载该环境的指令——否则它会找不到 `py-spy`。

别忘了在这个过程结束后手动释放资源分配。

## Crontab 模拟 {#crontab-emulation}

crontab 是最重要的 Unix 工具之一，对于调度各种作业必不可少。然而 SLURM 环境中通常没有它，因此必须模拟它的功能。具体做法如下。

为了这里的演示，我们将把 `$WORK/cron/` 用作基础目录，假设你已经导出了一个指向文件系统上某个位置的 `WORK` 环境变量——如果你使用 Bash，可以在 `~/.bash_profile` 里设置；如果使用的是别的 shell，就在对应的启动配置文件中设置。


### 1. 自我延续的调度作业 {#1-a-self-perpetuating-scheduler-job}

我们将使用 `$WORK/cron/scheduler` 目录存放调度器作业，`$WORK/cron/cron.daily` 存放每日作业，`$WORK/cron/cron.hourly` 存放每小时作业：

```bash
$ mkdir -p $WORK/cron/scheduler
$ mkdir -p $WORK/cron/cron.daily
$ mkdir -p $WORK/cron/cron.hourly
```

现在把这两个 slurm 脚本复制到 `$WORK/cron/scheduler` 中：

- [cron-daily.slurm](cron-daily.slurm)
- [cron-hourly.slurm](cron-hourly.slurm)

前提是先编辑好这两个脚本，使其符合你所在环境的账户和分区信息。

现在你可以启动 crontab 调度作业了：

```bash
$ cd $WORK/cron/scheduler
$ sbatch cron-hourly.slurm
$ sbatch cron-daily.slurm
```

就这样，这些作业此后会自我延续（self-perpetuate），通常你不需要再管它们，除非发生了使 SLURM 丢失所有作业的事件。


### 2. 每日与每小时的定时作业 {#2-daily-and-hourly-cronjobs}

现在，只要你想让某个作业每天运行一次，只需创建一个 slurm 作业并把它放进 `$WORK/cron/cron.daily` 目录即可。

下面是一个示例作业，它每天运行一次，用 `updatedb` 重建 `$WORK` 的文件索引（在当前发行版上，这通常是 `plocate` 的 `updatedb`；较旧的系统用的是 `mlocate`）：
```bash
$ cat $WORK/cron/cron.daily/mlocate-update.slurm
#!/bin/bash
#SBATCH --job-name=mlocate-update    # 作业名
#SBATCH --ntasks=1                   # MP 任务数量
#SBATCH --nodes=1
#SBATCH --hint=nomultithread         # 我们得到的是物理核心而非逻辑核心
#SBATCH --time=1:00:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=PARTITION     # 请修改
#SBATCH --account=GROUP@PARTITION # 请修改

set -e
date
echo "updating file index db"
/usr/bin/updatedb -o $WORK/lib/mlocate/work.db -U $WORK --require-visibility 0
```

这会为 `$WORK` 下的文件建立索引，之后你可以用下面的命令快速查询：
```
/usr/bin/locate -d $WORK/lib/mlocate/work.db pattern
```

要停止运行这个作业，只需把它移出 `$WORK/cron/cron.daily` 目录。

同样的原则适用于放进 `$WORK/cron/cron.hourly` 目录的作业。它们适合用来每小时运行某些任务。

请注意，这套 crontab 实现在时间精度上是近似的：由于 SLURM 调度中的各种延迟，它们只会大约每小时、每天运行一次。如果有必要，你可以重新改写这些脚本，要求 SLURM 在更精确的时间启动任务，但大多数时候，刚才介绍的方法就足够好了。

此外，你还可以自己编写变体，以满足项目的特定需求，例如每 30 分钟或每 12 小时运行一次的作业。


### 3. 清理 {#3-cleanup}

最后，由于每个 cron 启动作业都会留下一个日志文件（如果由于某种原因出了问题，这些日志反而很有用），你会想创建一个定时作业来清理这些日志。否则你可能会耗尽 inode——这些日志文件很小，但数量可能有几万个。

你可以在每日作业中使用类似这样的命令。

```bash
find $WORK/cron -name "*.out" -mtime +7 -exec rm -f {} +
```
请注意，它被设置为只删除早于 7 天的文件，以防你需要用最新的日志做诊断。


### 注意事项 {#nuances}

调度器以启动 SLURM cron 调度作业的用户的 Unix 权限运行，该 cron 作业启动的所有其他 SLURM 脚本也是如此。

## 自我延续的 SLURM 作业 {#self-perpetuating-slurm-jobs}

[构建调度器](#1-a-self-perpetuating-scheduler-job)一节中用到的同一套做法，也可以用来创建独立运行的自我延续作业。

例如：

```bash
#!/bin/bash
#SBATCH --job-name=watchdog          # 作业名
#SBATCH --ntasks=1                   # MP 任务数量
#SBATCH --nodes=1
#SBATCH --time=0:30:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#SBATCH --partition=PARTITION        # 请修改

# 先确保在 1 小时后重启自身
RUN_FREQUENCY_IN_HOURS=1
sbatch --begin=now+${RUN_FREQUENCY_IN_HOURS}hour watchdog.slurm

... 在这里执行看门狗的工作 ...
```
然后只启动它一次：
```bash
sbatch watchdog.slurm
```
这个作业随后会立刻把自己安排到启动时间之后 1 小时运行，然后执行正常的作业工作。无论作业的其余部分成功还是失败，它都会继续以大约每小时一次的频率重新调度自己。由于调度器启动作业的开销以及节点可用性问题，这个时间并不精确。但只要至少有一个空闲节点可用，并且作业本身能很快完成，那么以近似频率运行的要求就应该是够用的。

由于大多数 SLURM 环境除了昂贵的 GPU 节点之外，还提供便宜得多的纯 CPU 节点，对于不需要 GPU 的作业，你应该选择纯 CPU 的 SLURM 分区。


## 克服 SLURM 作业缺少组所有权的问题 {#overcoming-the-lack-of-group-slurm-job-ownership}

SLURM 运行在 Unix 上，但令人意外的是，它的设计者没有把组所有权这一概念应用到 SLURM 作业上。所以，如果你的团队中某位成员启动了一个包含 10 个作业、每个运行 20 小时的作业数组，然后就度假去了——万一出了问题，除非你有 `sudo` 权限，否则现在你对这些作业无能为力，没法把它们停掉。

我至今没有弄清楚为什么会这样，但到目前为止我们一直在使用一种终止开关（kill switch）的变通方案。你需要把它实现到自己的框架里。例如，可以看看它在 [Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L104)（Meg-DS）中是如何实现的。程序会轮询一个在启动时就已配置好的文件系统路径，如果在其中发现了文件，就退出。

因此，如果我们用 `--kill-switch-path $WORK/tmp/training17-kill-switch` 启动 Meg-DS，之后任何时候需要杀掉这个 SLURM 作业，只需简单地执行：

```bash
touch $WORK/tmp/training17-kill-switch
```
下一次程序检查这个文件时，它就会检测到这一事件并主动退出。如果你有一个作业数组，那就得等到每个作业都启动、检测到终止开关并退出为止。

当然，在把作业都停下来之后，别忘了把这个文件删掉。
```bash
rm $WORK/tmp/training17-kill-switch
```

不过，这并不总能奏效。如果作业挂起了，它就永远不会走到检查终止开关的那一步，此时唯一的解决办法是联系系统管理员帮你杀掉这个作业。有时候，如果只是简单的挂起，pytorch 的分布式设置通常会在预设的 10 分钟超时时间之后自动退出，但这并不总是管用。


## 如何在 SLURM 作业被抢占时优雅退出 {#how-to-gracefully-exit-on-slurm-job-preemption}

优雅处理基于时间和基于 QoS 的 SLURM 抢占有好几种方法，这些方法在[处理强制作业抢占](../../training/fault-tolerance/README.md#dealing-with-forced-job-preemption)一节中有详细讲解。


## 一个作业用了多少块 GPU {#how-many-gpus-a-job-uses}

要弄清一个已在运行的作业使用了多少块 GPU，可以解析 `show job -d` 输出中的 `JOB_GRES=gpu:` 条目。例如，如果作业是用下面的命令启动的：

```bash
srun --pty --partition=dev --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --time=8:00:00 bash
```
也就是说我们分配了 16 块 GPU，现在可以通过下面的方式以编程方式把这个数字取回来：

```bash
$ TOTAL_JOB_GPUS=$(scontrol show job -d $SLURM_JOBID | perl -ne 'm|JOB_GRES=gpu:(\d+)| && print $1')
$ echo $TOTAL_JOB_GPUS
16
```

如果运行命令的 shell 中没有设置 `$SLURM_JOBID`，就把它替换成 SLURM 作业 id（见 [`squeue`](#show-jobs)）。


## 作业运行了多长时间 {#how-long-did-the-job-take-to-run}

通常 `squeue` 会显示当前正在运行的作业已经运行了多久；而要查看一个作业结束时总共运行了多长时间，你需要知道它的作业 id，然后可以像这样查询：

```bash
$ sacct -j 22171 --format=JobID,JobName,State,Elapsed
JobID           JobName      State    Elapsed
------------ ---------- ---------- ----------
22171          example   COMPLETED   00:01:49
```

由此我们知道这个作业在不到 2 分钟内就跑完了。

## FairShare {#fairshare}

许多 SLURM 集群使用 FairShare 机制：某人使用集群越多，其运行作业获得的优先级就越低；如果集群启用了抢占，他们的作业也更可能被抢占。

要查看你的 FairShare 分数，运行：
```bash
sshare
```

示例：

```
Account                    User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare
-------------------- ---------- ---------- ----------- ----------- ------------- ----------
root                                          0.000000   711506073      1.000000
  all                                     1    0.500000   711506073      1.000000
   all                      stas          1    0.022727    14106989      0.019827   0.288889
```

如果你的 FairShare 分数高于 0.5，说明你使用集群的量低于分配给你的份额；如果低于 0.5，说明你使用的量超过了分配给你的份额。

随着时间的推移，这个分数会被衰减。所以如果你之前的分数很低，并且此后一直很少使用集群，那么你的分数会随时间上升。

要查看某个特定用户的分数：
```bash
sshare -u username
```

要查看所有人的分数，并按 FairShare 排序：
```bash
sshare --all | sort -nk7 -r
```

这才是最重要的输出，因为只看你自己的分数本身并没有意义，重要的是你的分数相对于所有其他用户的高低。任何分数比你高的人，其作业都有更大的机会最先被调度到资源，其作业被抢占的可能性也更小。

除了 FairShare，优先级通常基于多个指标的组合来配置，一般包括作业在队列中等待的时间、作业规模、服务质量（QOS）设置、分区的具体配置等。具体细节取决于你的系统管理员如何配置 slurm。
