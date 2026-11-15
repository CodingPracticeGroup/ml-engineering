# 存储：文件系统与 IO {#storage-file-systems-and-io}

## 机器学习的 IO 需求 {#machine-learning-io-needs}

对于训练负载，有 3 种截然不同的 IO 需求：

1. 你需要能够快速喂饱 DataLoader ——（极快的读，不在乎是否快写）—— 要求能持续承载数小时乃至数天的负载
2. 你需要能够快速写入检查点（checkpoint）——（极快的写，较快地读，因为你只会恢复几次）—— 要求突发式写入 —— 你希望写入快到不会长时间阻塞训练（除非你使用某种 CPU 卸载来尽快解除对训练的阻塞）
3. 你需要能够加载并维护你的代码库 ——（读写都是中等速度即可）—— 这还要求它是共享的，因为你要所有节点看到同一份代码库 —— 由于它只在启动或恢复时发生，所以频率很低

可以看到，这三者在速度和可持续负载上的要求非常不同，因此理想情况下你应当有 3 个不同的文件系统，分别针对各自的使用场景做优化。

当然，如果你预算无限，那就买一套读极快、写也极快、还能连续好几天不掉速的存储。但对我们大多数人来说这做不到，所以配置 2 到 3 种不同类型的分区、从而少花钱，是更明智的选择。

对于推理，主要考虑的是启动时间，以便尽快开始提供推理服务：

1. 快速读取检查点
2. 快速加载代码库

对于批量推理，你还会额外需要：

1. 读取提示词（prompt）
2. 写出生成的输出

如果这些操作与生成过程异步执行，那么这部分 IO 开销可以被完全隐藏，因为通常生成过程要比"读取生成所依据的文本、再把生成的文本写回磁盘"慢得多。

如果你把 KV 缓存卸载（offload）到磁盘，那就是推理中另一个重要的 IO 使用场景。你需要能够快速地寻址（seek）到想要的 KV 缓存索引；至于把它写出去，如果写入是异步完成的，就没那么关键了。


## 术语表 {#glossary}

- IOPS：每秒输入/输出操作数（Input/Output Operations Per Second）
- NSD：网络共享磁盘（Network Shared Disk）


## 概念 {#concepts}

以下是几个你很可能需要熟悉的存储相关关键概念：

### 队列深度 {#queue-depth}

**队列深度**（queue depth，又称 **IO 深度**）是存储设备控制器上同一时刻能够排队的 IO 请求数量。如果发出的 IO 请求超过了控制器能排队的数量，操作系统通常会把多出来的请求放进自己的队列。

在 Linux 上，本地块设备的队列深度通常由内核预先配置好。例如，想查看为 `/dev/sda` 设置的最大队列深度，可以执行 `cat /sys/block/sda/queue/nr_requests`。想看本地设备当前的队列深度，运行 `iostat -x` 并观察 `aqu-sz` 列。（运行 `apt install sysstat` 可以获得 `iostat`。）

存储设备会同时服务多个请求——NVMe 盘同时从许多闪存芯片读取，网络文件系统（通常）同时从许多存储服务器读取。可以把这些并发的处理单元看作设备的"工人"。只要正在处理的请求数少于工人数量，每个请求就直接交给一个空闲的工人，因此延迟保持不变，而吞吐量随每个新增请求而上升。相反，如果你一次只发出一个请求，那么设备每次完成一个请求后都会闲置下来，等应用程序再发出下一个请求。

一旦请求数超过工人数量，吞吐量就达到上限，多出来的请求进入排队，于是延迟随队列深度增长。

### 直接 IO 与缓冲 IO {#direct-vs-buffered-io}

**直接** IO（direct IO）指绕过操作系统缓存缓冲区的 IO。它对应 [`open(2)`](https://man7.org/linux/man-pages/man2/open.2.html) 系统调用中的 `O_DIRECT` 标志。

与之相反的是**缓冲** IO（buffered IO），由于缓存通常会让事情变快，它一般是大多数应用程序做 IO 的默认方式。

运行 IO 基准测试时，关闭缓存/缓冲非常关键，否则基准测试的结果很可能无效。在实际使用中，你通常不会连续几百次读写同一个文件。因此，如果基准测试工具提供了直接（direct）模式的开关，你多半应该把它打开。

在某些情况下，以 `O_DIRECT` 打开文件实际上有助于消除延迟。例如，如果训练程序把日志写到日志文件（尤其是在慢速的共享文件系统上），当应用程序的缓冲和文件系统的缓冲都挡在中间时，你可能要过很多秒才能看到日志。写入方用 `O_DIRECT` 打开日志文件，就能让读取方更早看到写入的日志行。

在把 `O_DIRECT` 用于你的工作流之前，值得先了解它不保证什么：

- 它不提供持久性（durability）。它"会尽力同步传输数据，但并不给出 `O_SYNC` 标志那样的保证——即数据与必要的元数据都已传输完成"——如果你需要这些字节在崩溃之后仍然保存，你应该用 `fsync`/`fdatasync`，或者在 `O_DIRECT` 之外再加上 `O_SYNC`。
- 它可能对缓冲区的长度、缓冲区的内存地址以及文件偏移施加对齐限制，具体要求因文件系统和内核版本而异。一次未对齐的 IO 可能以 `EINVAL` 失败，也可能悄悄退化为缓冲 IO——基准测试就这样悄无声息地不再度量你以为它在度量的东西，这种方式非常隐蔽。自 Linux 6.1 起，`statx(2)` 配合 `STATX_DIOALIGN` 会告诉你实际的要求。
- 在同一个文件上混用 `O_DIRECT` 和普通缓冲 IO 是被明确不推荐的，尤其是在相互重叠的区域——一个用 `O_DIRECT` 写入、一个用 `tail -f` 读取，正是这样一种组合。

不过，遇到日志问题，先考虑应用程序自己的缓冲——挡在日志行前面的通常是这一层，而改它的代价要低得多：Python 用 `python -u` 或 `PYTHONUNBUFFERED=1`，在 `print` 调用里加 `flush=True`，对管道中没有自带刷新选项的 C stdio 程序（如 `awk` 或 `cut`）用 `stdbuf -oL`。`O_DIRECT` 对此毫无作用，因为它绕过的是内核的页缓存（page cache），而不是你进程自己的缓冲区——一行日志还留在应用程序的缓冲区里时，无论文件用什么标志打开，它都根本还没有被写出。

这里做一个小演示。`awk` 是一个普通的 C stdio 程序，正是 `stdbuf` 能修复的那一类程序——末尾的 `cat` 只是为了让 `awk` 的标准输出变成管道而不是终端：

```bash
{ echo 1; sleep 3; echo 2; } | awk '{print}' | cat
```

3 秒后两行一起出现。`awk` 并不是拒绝打印，它只是打印到了自己的缓冲区——当 stdout 不是终端时，stdio 会从行缓冲切换为几千字节的块缓冲，只有当缓冲区写满或程序退出时才写出。加上 `stdbuf -oL`，`1` 就会立即出现：

```bash
{ echo 1; sleep 3; echo 2; } | stdbuf -oL awk '{print}' | cat
```

反过来，去掉 `| cat` 让 `awk` 直接面向你的终端，stdio 默认就是行缓冲，因此两个版本表现完全一致，也没什么需要修复——只有当你把输出通过管道交给另一个程序、或重定向到日志文件时，缓冲才会开始碍事，而那正是你无法亲眼看到它发生的时刻。

`stdbuf` 预设的是 C stdio 在启动时读取的缓冲模式，所以它只对把这一决定交给 stdio 的程序起作用。`perl` 和 `python` 都自行管理缓冲，会完全忽略它，Go 编译出的二进制也是如此——对这些程序，调节的旋钮必须放在程序内部：perl 用 `$|=1`，python 用 `-u` 或 `PYTHONUNBUFFERED=1`。


### 同步 IO 与异步 IO {#synchronous-vs-asynchronous-io}

在同步 IO 中，客户端发出一个 IO 请求，要等它完成之后，才向同一个目标设备发出下一个 IO 请求。

在异步 IO 中，客户端可以一个接一个地发出多个 IO 请求，而不必等待前面的任何一个先完成。这要求目标设备能够[对多个 IO 请求排队](#queue-depth)。


### 顺序访问与随机访问 IO {#sequential-vs-random-access-io}

**顺序访问** IO 指一个接一个地按顺序读取数据块（想象播放电影）。下面是一些例子：

- 一次性读取或写入模型的检查点文件
- 加载一个 python 程序
- 安装一个软件包

**随机访问** IO 指随机地访问文件的某一部分。下面是一些例子：

- 数据库查询
- 以随机方式从预处理好的数据集中读取样本
- 用 `seek` 在文件中来回移动


### mmap 与数据集的顺序读取 {#mmap-vs-sequential-dataset-reads}

一个看起来像单个大文件的数据集，最终仍可能在做随机 IO。Hugging Face `datasets` 把数据保存在 [Apache Arrow](https://huggingface.co/docs/datasets/en/about_arrow) 文件中——这是一种二进制列式格式，其在磁盘上的字节布局与在内存中的使用方式一致，因此可以直接就地读取某一行，无需解析步骤。正因如此，内存映射（memory-mapping）才成为默认方式（`load_from_disk`，或不带 `keep_in_memory` 的 `load_dataset`）：文件被映射进地址空间，访问一行就变成了缺页中断（page fault），而不是一次 `read` 调用。

脚注：这种映射在 `top` 中显示出的常驻内存并不是内存泄漏——操作系统可以回收未被引用的页——参见 [CPU 内存](../compute/cpu-memory/README.md#things-to-know)。

如果你统计 `read()` 系统调用次数——用 `strace -c`，或者 grep 跟踪记录——被映射的数据集看起来几乎处于空闲状态。下面每次映射读取整个 1GiB 只发出了 2 次 `read` 调用；数据确实搬运了，只不过体现为主缺页中断（major page fault），以及 `/proc/self/io` 中的 `read_bytes` 计数。

以上数据于 2026-09 使用 `datasets` 4.0.0 测得：1GiB 文本，切为 32768 行、每行 32KiB，存放在三个 Arrow 分片（shard）中；每次运行前用 `posix_fadvise(..., POSIX_FADV_DONTNEED)` 丢弃页缓存。第一行是基准下限——用普通的每次 1MiB 的 `read()` 调用把同样这三个文件从头读到尾。另外两行走的是映射后的数据集：先是 `for row in ds`，然后按打乱后的顺序访问 `ds[i]`。

| 这 1GiB 数据的读取方式            | 本地 NVMe | 减速幅度 | Lustre | 减速幅度 |
| :------------------------------ | ---------: | -------: | -----: | -------: |
| 顺序 `read()` 读取整个文件        |      0.14s |       0% |  1.87s |       0% |
| mmap，按顺序逐行访问              |      0.53s |     279% |  3.11s |      66% |
| mmap，按打乱顺序逐行访问          |      0.78s |     457% |  94.2s |   4_937% |

- 减速幅度列的值是同一文件系统内的 `t / t_sequential - 1`。
- 本表由 [`mmap-io-bench.py`](./mmap-io-bench.py) 生成——运行前请先编辑脚本开头的两个路径。本表中的基准测试文件系统是 Lustre。

这三行对应训练流水线接触 Arrow 分片的三种典型方式：把文件整体读一遍、按顺序遍历映射后的数据集、按打乱后的训练顺序索引行。把你的目标作业对号入座——下面的条目会说明什么情况下这种减速可以接受，什么时候需要换一条读取路径，或者把分片放到节点本地磁盘而不是网络文件系统。

<!---->

1. **按顺序消费一个分片**——校验和、格式转换、单趟分词。用每次大约 1MiB 的普通 `read` 读取文件，不要做映射。这就是为什么仅仅把数据集存成大文件是不够的：每个读取者还必须以大块顺序读取的方式消费它们，才能获得高性能。
2. **在映射后的数据集上跑 `for row in ds`**——在本地 NVMe 上，映射带来 279% 的开销，但半秒钟仍能读完一个 GiB，所以可以保持现状。在 Lustre 上它是 66%，而且每一个页都要走网络，所以如果这个循环是在为训练供数据，先把分片复制到节点本地磁盘。
3. **在映射后的数据集上按打乱顺序访问 `ds[i]`**——这就是普通的打乱训练。在 Lustre 上，每 GiB 要 94s，是顺序读取相同数据的 50 倍。不要把数据集指向网络文件系统。把该切分（split）预先放到节点本地 NVMe 上再映射；或者预先打乱并按 epoch 组织成分片，再按顺序读取这些分片；或者流式读取、在缓冲区内部打乱。`keep_in_memory=True` 也能消除缺页中断，但代价是换成内存占用，而且每个 DataLoader 工作进程都要再各自付一遍这份开销。

脚注：在循环之前先顺序读完这些分片——`cat shard > /dev/null`，或者一次 `cp`——会把它们留在页缓存中，于是打乱访问的情形改为针对内存发生缺页：在 Lustre 上，冷启动的 98.24s 变成了 1.86s 的预读加上 1.03s 的循环。这只在分片能容纳进页缓存时成立，因此它只适用于能放进内存的切分，不能替代放到本地 NVMe 上的做法。映射提示也帮不上忙，而且无论如何都通过 `datasets` 用不到——[`MAP_POPULATE`](https://man7.org/linux/man-pages/man2/mmap.2.html) 会为整个映射预先缺页，[`MADV_SEQUENTIAL`](https://man7.org/linux/man-pages/man2/madvise.2.html) 只是加大预读，而 `pyarrow` 的 `memory_map` 两者都不传。


### 上报的文件大小不正确 {#misreported-file-size}

我注意到一些分布式文件系统（如 Lustre）在文件被卸载（offload）到冷存储、尚未"重新加载（rehydrate）"时，可能会报告不正确的文件大小。我在 Weka 或 GPFS 上没有遇到过这个问题。一个称职的分布式文件系统客户端应当始终报告真实的文件大小，即使文件内容已被卸载，并在文件被读取时自动把它重新加载回来。

如果你倒霉遇到了这样有缺陷的文件系统客户端，可以用 `du --apparent-size` 大致了解真实的文件大小，但要注意：由于碎片化、文件稀疏（sparse）等原因，它也可能高估大小。`df` 仍然会上报错误的文件大小，因为它没有类似 `du` 的那个选项。

如果需要强制触发重新加载，可以运行类似这样的命令：

```bash
find /mountpoint/ -type f -exec cat {} >/dev/null \;
```
之后 `du` 和 `df` 都会报告正确的文件大小，不过如果你有几百 GB 的数据，上面这条命令可能要跑很长时间。

只要可能，就不要使用连"报告正确文件大小"这种基本需求都满足不了的文件系统，因为一旦发生这种情况，你可能完全没有意识到你的分区已经接近写满。例如，它可能在已经用了 95% 时报告只用了 5%。


## 选择哪种文件系统 {#which-file-system-to-choose}

**分布式并行文件系统是最快的方案**

在有数百到数千个客户端可以同时访问共享存储的场景下，分布式并行文件系统能显著提升性能。它们在减少热点（hotspot，即某些数据块被访问的频率远高于其他数据）方面也很有效。

我亲身经历过的 3 种性能出色的并行文件系统是：

- [GPFS](https://en.wikipedia.org/wiki/GPFS)（IBM），最近更名为 IBM Storage Scale，
    在此之前它曾叫 IBM Spectrum Scale。

- [WekaIO](https://www.weka.io/)
- [Lustre FS](https://www.lustre.org/)（开源）（[Wiki](https://wiki.lustre.org/Main_Page)）

这些方案已经有 20 多年的历史。它们符合 POSIX 标准。它们也不是随随便便就能搭起来的——你得先另建一个集群，配备多台专门服务于这些文件系统的 CPU-only VM，之后才能挂载它们。相比之下，云服务商提供的较弱的"内置"方案只需要回答几屏问题就能开通。而在创建存储集群时，为哪些功能选择哪些 VM 也是一门学问。例如，这里有一份[GCP 上的 Lustre 指南](https://docs.cloud.google.com/architecture/parallel-file-systems-for-hpc#overview_of_lustre_and_exascaler_cloud)。

案例：2021 年，在（法国的）JeanZay HPC，我们用 384 个进程并行地在 40 秒内写完了 2.3TB 的检查点！这快得不可思议——用的是架在 NVMe 盘上的 GPFS。

NASA 的集群有[一长串关于使用 Lustre 的注意事项](https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html)。

GPFS 的一些非常实用的优点：

- 如果你有大量小文件，inode 很容易用完（用 `df -i` 检查）。GPFS 5.x 永远不会用完 inode，它会按需动态创建更多 inode
- GPFS 没有 Lustre 的那种问题：某个子盘满了又没及时再平衡，整个文件系统就会在 80% 时就报磁盘空间不足——你可以可靠地用满已分配存储的 100%。
- GPFS 不使用中央元数据服务器（或由它们组成的集群），而中央元数据服务器在处理小文件时常常成为瓶颈。和元数据一样，GPFS 把元数据像数据一样交由存储集群中的每个节点处理。
- GPFS 自带原生的 NSD 客户端，它优于通用的 NFS 客户端，不过两者都可以配合 GPFS 使用。
- 可以构建多层级（multi-tier）系统。例如，Tier 1 通常由 NVMe 盘构成，Tier 2 通常使用某种云存储系统。当 Tier 1 的容量偏低时，一段时间内未被访问的文件会被自动移动到云存储。例如，你的 Tier 1 可以是 100TB，Tier 2 可以是 1PB。这种做法能省下很多钱，因为 1PB 的云存储比 1PB 的 NVMe 盘便宜得多。
- 数据保护可以采用各种 RAID 方式。通常使用条带化（striping）来节省成本。

Weka 在功能和性能上与 GPFS 相当接近。主要的差别在于你能与两家供应商谈下来的许可费用。你的很大一部分成本会来自运行这套系统所需的 VM 费用——例如，如果你有很多小文件，就会希望有很多台 VM 来快速处理元数据。

其他我尚无直接使用经验的并行文件系统：

- [BeeGFS](https://www.beegfs.io/)
- [DAOS](https://docs.daos.io/latest/)（Distributed Asynchronous Object Storage，分布式异步对象存储）（Intel）
- [NetApp](https://www.netapp.com)
- [VAST](https://www.vastdata.com/)

大多数云服务商都提供其中至少一种实现，但不是全部。如果你的云服务商这几种都不提供，又没有足够快的替代方案来满足你的需求，你就应当重新考虑。

**勉强可用的方案**

[各家云服务商](#cloud-shared-storage-solutions)提供了很多勉强可用的方案。在你选定任何一家之前，认真对它们跑基准测试。这类方案处理大文件通常还算不错，处理小文件就差得多。

案例：截至 2024-01，在 GCP 的 Zonal FileStore over NFS 方案上，执行 `python -c "import torch"` 需要 20 秒，极其缓慢！文件被缓存之后则大约需要 2 秒。创建一个装有若干预构建 python 包的 conda 环境，动辄要花 20-30 分钟！我们最初采用的这个方案非常痛苦，对我们的工作起到了反作用。任何有大量 python 包和 conda 环境的人都会受到影响。当然，GCP 也提供快得多的方案。

## 远程文件系统客户端 {#remote-file-system-clients}

你需要选择一个客户端，用它把文件系统连接到你的 VM 上。

最常见的选择是：[NFS](https://en.wikipedia.org/wiki/Network_File_System)——它已经存在了 40 年。它会引入额外的开销并使速度变慢。所以，如果你的 VM 支持某个原生客户端，使用它通常会比用 NFS 获得更高的整体性能。例如，GPFS 自带的 [NSD](https://www.ibm.com/docs/en/linux-on-systems?topic=configurations-network-shared-disk-nsd) 客户端就优于 NFS。

## 文件块大小 {#file-block-size}

如果你使用的文件系统块大小为 16MiB，而你的文件平均大小只有 16KiB，你占用的磁盘空间就会是实际需要的 1,024 倍。例如，你会看到磁盘上已用空间为 100GiB，而实际数据只有 100MiB。

脚注：在 Linux 上，本地文件系统通常使用 4KiB 的块大小。

因此你往往会有两种截然不同的需求，需要两个针对不同需求优化的分区：

1. 数以千计到数以百万计的小文件——4-8KiB 块大小
2. 少量大文件——2-16MiB 块大小

案例：python 动辄产生几万个微小文件，这一点非常糟糕，如果你有很多 conda 环境，在某些情况下你很可能会耗尽 inode。在 JeanZay HPC，我们不得不申请一个专门的分区来安装所有 conda 环境，因为我们在普通的 GPFS 分区上总是把 inode 用光。我想问题在于那些 GPFS 分区配置的是 16MiB 的块大小，所以对于 4KiB 大小的文件来说，它们并不是合适的分区。

好消息是，现代方案开始引入动态块大小。例如，最新的 GPFS 支持子块（sub-block）。比如，你可以把 GPFS 配置为 2MiB 的块大小并配一个 8KiB 的子块，这样微小文件会被打包成子块存放，从而不会浪费太多磁盘空间。

## 分布式存储服务器与客户端的邻近关系 {#distributed-storage-servers-proximity-to-clients}

使用共享分布式存储的集群，其存储服务器应当放置在使用这些服务器的集群附近。如果运行存储服务器的 VM 相距很多跳（交换机），IO 延迟可能很高，交互式地使用存储会慢得让人抓狂。举个例子：当你运行 `du` 以及其他访问大量文件元数据的工具时，与元数据服务器之间的任何交互都会如此。

因此，如果你能施加影响，就要求云服务商把运行存储服务器的 CPU-only VM 在网络距离上尽可能分配到你的加速器 VM 附近。



## 云共享存储方案 {#cloud-shared-storage-solutions}

以下是各家云服务商提供的共享文件系统存储方案：

- [GCP](https://docs.cloud.google.com/architecture/parallel-file-systems-for-hpc)
- [Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/disks-shared)
- [AWS](https://aws.amazon.com/what-is/nas/#how-can-aws-help-with-storage-solutions--mxce5s)


## 本地存储优于云存储 {#local-storage-beats-cloud-storage}

虽然云存储更便宜，但"在训练时动态获取并处理训练数据流"这个整体思路问题非常多，围绕它可能出现一大堆问题。

把检查点动态卸载到云上也一样。

为数据加载准备足够的本地磁盘空间要好得多。

对于检查点保存，应当有足够的本地磁盘空间，以便快速、可靠地保存检查点，然后再用 crontab 作业或 Slurm 作业把它卸载到云上。始终在本地保留最近几个检查点，这样作业崩溃后可以快速恢复，因为为了恢复而等待从云端取回检查点是极其昂贵的。

案例：在 IDEFICS-80B 训练期间，我们别无选择，只能用云存储做数据加载，因为我们几乎没有本地存储，而那又是多模态数据，数据量有好几 TB。我们花了好几周想让这个方案变得稳健，但最后它还是不行。当时最大的问题是，很难跟踪 DataSampler 的 RNG（随机数生成器）状态，因为我们用的那个解决方案——唉——根本没打算管这件事。于是，很多花了大量时间做出来的数据被浪费（没有被使用），同时很多数据被重复使用，我们连一个全部由不重复数据组成的 epoch 都没能做到。

在某些情况下，人们能找到处理云端数据集的好方案，但就我个人而言，我还没有过顺利的体验，这也是我提倡使用本地存储的原因。如果你找到了一个出色的流式方案——能够正确恢复、不丢数据也不重复数据、并且不需要庞大的本地工作节点——那么它也许能用得不错。

让这种方式行得通的关键是数据布局。流式读取会从云对象存储中整个地拉取对象——S3 风格的存储桶（bucket），其 API 返回的是整个对象，而不是让你只读其中的一块——所以只有当数据集本身就是一组可以从头到尾顺序消费的大分片时，它才划算。这正是 [mmap 与数据集的顺序读取](#mmap-vs-sequential-dataset-reads) 一节中在共享文件系统上实测最快的那种数据形态，因此处于那种形态的数据集不需要在那里再存一份副本——存储桶可以直接为训练供数据。

这个整对象接口同时也是一道护栏，因为它使得在对象内部寻址或对对象做内存映射变得不可能。而把存储桶挂载成一个普通目录会把这道护栏拆掉：上一节在网络文件系统上（那个例子中是 Lustre）被测出比顺序读取慢 50 倍的随机访问与内存映射模式，又会重新开始工作——面对一个从来就不是为它们设计的后端。

## 注意：你往往只买到了所付费存储的 80% {#beware-that-youre-often-being-sold-only-80-of-the-storage-you-pay-for}

在计算节点上使用分布式共享存储时，有一个微妙的问题。由于构建大型文件系统所用的物理磁盘大多只有 0.3-2TB，任何一块物理盘都可能在整个存储写满之前就先写满。因此需要持续地进行再平衡，避免出现某块盘已经 99% 满、其他盘却只有 50% 的情况。而再平衡是代价高昂的操作，就像大多数编程语言的垃圾回收一样，它不会频繁发生。于是，如果你运行 `df` 看到报告 90% 已满，那么任何一个程序都很可能在任何时刻失败。

从与 IO 工程师们的交流来看，他们接受的现实（不知为何没有向客户传达）是：大型分布式存储中只有约 80% 是可靠的。

这意味着，如果你想要 100TB 可靠可用的云存储，实际上需要购买 125TB 的存储，因为其中的 80% 才是那 100TB。所以你要计划比按实际需求准备的量多付 25% 的费用。我不确定为什么客户要为技术的缺陷买单，但现实就是这样。

例如，GCP 声明只有 [89%](https://docs.cloud.google.com/filestore/docs/known-issues#capacity_errors_before_reaching_full_provisioned_capacity) 可以可靠使用，尽管就我而言，那里的存储已经不止一次在用到 83% 时就出错了。值得肯定的是，Google 至少把这披露为已知问题，尽管不是在人们购买存储的那个环节披露。也就是说——我们建议你比实际计划使用的量多买 12% 的存储，因为我们只能可靠地交付其中的 89%。

我还与提供 IBM Storage Scale（GPFS）托管服务的 [Sycomp](https://sycomp.com/) 工程师交谈过，据他们所说，GPFS 没有这个问题，整整 100% 都可以可靠使用。

另外，在某些配置下，如果你通过云服务商的 API 做备份（而不是直接在文件系统上操作），备份可能最终占用的是同一个分区，当然也会消耗磁盘空间，但你运行 `df` 时它不会显示真实的磁盘用量——它显示的用量可能不包含备份。

无论你选择哪种存储方案，都要向供应商询问有多少存储可以可靠使用，以免日后出现意外。
## 注意：有些云服务商的备份使用的正是被备份的那个分区 {#beware-that-on-some-cloud-providers-backups-use-the-same-partition-they-backup}

这让我觉得不可思议，但在有些服务商那里，当你用他们的工具给某个分区做备份时，备份会占用同一个分区的空间。而且其中一些服务商，你要等到磁盘空间耗尽时才会意识到发生了这种事，而实际上你只用了所分配分区的 30%。在这些服务商那里，运行 `df` 毫无意义，因为它报告的是空闲磁盘空间，但其中不包含任何备份。所以你完全不知道发生了什么。

如果你开始做备份，随后一切都失败了——所有进程都无法写入——而 `df` 却报告只用了 30%，现在你就知道这是为什么了。快照用的也是同一个分区。

所以，假设你买了一个 100TB 的分区，已经用掉了 95TB，现在想给它做备份——好吧，做不到——如果只剩下 5TB 的空间，就算能压缩，又能把 95TB 的数据放到哪里呢。

每当我发现更多具有这种反直觉行为的具体方案，我都会补充说明怎样才能看到真实的磁盘用量：

- [GCP FileStore](https://docs.cloud.google.com/filestore/docs/monitoring-instances#free-raw-capacity-percent)（但它对 Basic Tier 不起作用）


## 别忘了校验和 {#dont-forget-the-checksums}

向云端同步数据或从云端同步数据回来时，一定要先弄清你使用的工具是否会检查校验和（checksum），否则你可能会得到在传输过程中损坏的数据。有些工具会自动做校验，另一些则必须由你启用这个功能（因为它通常会带来额外的计算开销并拖慢传输）。慢一点没关系，但要安全。

这些通常是 MD5 和 SHA256 校验和。如果你的环境足够安全，通常 MD5 就足够了；如果你想要额外的安全性，就用 SHA256 校验。


## 基准测试 {#benchmarks}

时间就是金钱，无论指的是开发者的时间还是模型的训练时间都是如此，所以存储 IO 绝不能成为你的人力和计算工作流中的瓶颈，这一点至关重要。

接下来几节将讨论各种方法，用来判断候选的存储方案能否满足你的工作需求。

### 指标 {#metrics}

人们通常关心的三个主要存储 IO 指标是：

1. [吞吐量](https://en.wikipedia.org/wiki/Network_throughput) 或带宽（单位：字节每秒——可以是 MBps、GBps 等）
2. [IOPS](https://en.wikipedia.org/wiki/IOPS)（一个系统每秒能执行的输入/输出操作次数）
3. [延迟](https://en.wikipedia.org/wiki/Latency_(engineering))（msecs 或 usecs）

<!---->

- *IOPS* 衡量的是给定的存储设备或集群每秒能执行多少次输入和/或输出操作。通常读 IOPS 和写 IOPS 不会相同，而且对很多系统来说，它还取决于操作是顺序的还是随机的。所以一个存储系统会有 4 个不同的 IOPS 数值：

<!---->

1. 随机读的 IOPS
2. 随机写的 IOPS
3. 顺序读的 IOPS
4. 顺序写的 IOPS

<!---->

- *吞吐量* 指的是每秒能处理多少数据。

IOPS 与吞吐量的对比

- 处理小文件时，高 IOPS 很重要。
- 处理大文件时，高吞吐量很重要。

IOPS 与吞吐量通过块大小（block size）相关联：`Throughput = IOPS * block_size`

因此，在 IOPS 固定的情况下——系统单次能读写的块越大，吞吐量就越大。

而由于 IOPS 有 4 类，相应地也有 4 个与之匹配的吞吐量数值。

*延迟*：指从发出数据传输指令的那一刻，到收到该指令的响应为止的延迟。

通常，数据包要经过的距离越长（交换机、中继、实际物理距离），延迟就越大。

所以，如果你有一块本地 NVME 盘，你的读或写延迟会比读写一台位于另一个大洲的存储设备短得多。



### fio {#fio}

[fio - Flexible I/O tester](https://fio.readthedocs.io/en/latest/) 是一款常用的 IO 基准测试工具，相对易于操作。它有很多选项，几乎可以模拟任何类型的负载，并能给出非常详细的性能报告。

首先用 `apt install fio` 安装 `fio`，或者用你的包管理器对应的方式安装。

下面是一个读基准测试的示例：

```bash
base_path=/path/to/partition/
fio --ioengine=libaio --filesize=16k --ramp_time=2s --time_based --runtime=3m --numjobs=16 \
--direct=1 --verify=0 --randrepeat=0 --group_reporting --unlink=1 --directory=$base_path  \
--name=read-test --blocksize=4k --iodepth=64 --readwrite=read
```

这段命令会让 16 个并发读线程运行 3 分钟。该基准测试以顺序读取的方式，使用 4KiB 的块大小（大多数操作系统的典型值）、16KiB 的文件大小（大多数 Python 文件的常见大小），并采用[非缓冲 IO](#direct-vs-buffered-io)。因此，这组特定的参数能够很好地近似反映：在 16 个并发进程下导入 Python 模块能有多快。

案例：在一套 NFS 环境中，我们第一次运行 `python -c "import torch"` 花了 20 秒，比在普通 NVME 盘上跑同样的测试慢了大约 5-10 倍。诚然，文件一旦被缓存，加载就快得多，但由于一切都是慢的，开发过程非常痛苦。

值得一读：[Fio Output Explained](https://tobert.github.io/post/2014-04-17-fio-output-explained.html) - 这是一篇旧文，但依然是好东西 - 如果你有更新的文章，请给我发个链接或 PR。

重要：如果你不使用 `--unlink=1` 参数，请务必在不同基准测试之间删除 `fio` 的工作文件——不这样做会导致报告严重失真，因为 `fio` 会复用它在别的基准测试中准备的文件，而基准测试的参数一旦改变，这些文件是不应被复用的。据说这种复用是 `fio` 的一项特性，但在我看来这就是个 bug：我不知道这个细节，因此得到了大量无效报告，并且过了一阵才意识到它们是错的。

回到基准测试——参数需要随你希望提速的那类 IO 操作而变化——是在做大量 pip install，还是在 512 个进程上写 checkpoint，还是从 parquet 文件做随机读取——每个基准测试都要做相应调整，才能测量到正确的目标。

一开始我是手动挑出我关心的数据，后来我把它自动化了，做出了 [fio-scan](./fio-scan) 基准测试：它会在 16KiB、1MiB 和 1GiB 的文件大小下各运行一对读/写基准测试，使用固定的 4KiB 块大小（共 6 个基准测试）。它使用一个辅助脚本 [fio-json-extract.py](./fio-json-extract.py) 解析日志文件，提取平均延迟、带宽和 iops，并以格式整齐的 markdown 表格报告出来。

下面是运行方法：
```bash
git clone https://github.com/stas00/ml-engineering/
cd ml-engineering
cd storage

path_to_test=/path/to/partition/to/test
./fio-scan $path_to_test
```
请调整 `path_to_test`，使它指向你想做基准测试的分区路径。

注意：`fio-scan` 会调用 `python ./fio-json-extract.py`，所以如果它失败了，请检查 `python` 实际指向哪里——很多系统上只有 `python3`。请编辑 `fio-scan`，让它指向正确的解释器。

下面是这个 IO 扫描在我的 Samsung SSD 980 PRO 2TB NVME 盘上得到的示例结果，于 2023-12-20 针对 `/mnt/nvme0` 分区运行：

* filesize=16k 读取

| 延迟 msec | 带宽 MiBps | IOPS   | 作业数 |
| -------: | -------: | -----: | ---: |
|      4.0 |   1006.3 | 257614 |   16 |

* filesize=16k 写入

| 延迟 msec | 带宽 MiBps | IOPS   | 作业数 |
| -------: | -------: | -----: | ---: |
|      3.2 |   1239.1 | 317200 |   16 |

* filesize=1m 读取

| 延迟 msec | 带宽 MiBps | IOPS   | 作业数 |
| -------: | -------: | -----: | ---: |
|      1.7 |   2400.1 | 614419 |   16 |

* filesize=1m 写入

| 延迟 msec | 带宽 MiBps | IOPS   | 作业数 |
| -------: | -------: | -----: | ---: |
|      2.1 |   1940.5 | 496765 |   16 |

* filesize=1g 读取

| 延迟 msec | 带宽 MiBps | IOPS   | 作业数 |
| -------: | -------: | -----: | ---: |
|      1.4 |   2762.0 | 707062 |   16 |

* filesize=1g 写入

| 延迟 msec | 带宽 MiBps | IOPS   | 作业数 |
| -------: | -------: | -----: | ---: |
|      2.1 |   1943.9 | 497638 |   16 |

可以看到，这是一块非常快的 PCIe 4.0 NVMe 盘，如果你想拿它作为基线，去和某个网络共享文件系统之类的东西作对比，它是合适的参照。


### 使用感知类 IO 基准测试 {#usability-perception-io-benchmarks}

除了那些设计得当的性能基准测试——它们会给你一些你可能看得懂、也可能看不懂的数字——还有一种感知基准测试，也就是：某种功能或服务用起来感觉如何。例如，打开一个网站时，是不是感觉网页加载太慢？使用视频服务时，视频是不是要等很久才开始播放，是不是每隔几秒就停下来缓冲？

所以对文件系统来说，问题非常简单——安装或启动一个程序，是不是感觉太久了？我们很多人整天生活在 Python 世界里，而 Python 以拥有成千上万个小文件著称，这些文件通常被安装进虚拟环境，截至 2026-08，[conda](https://www.anaconda.com/download) 仍是许多人的选择。

在某个环境中，我们注意到开发人员在共享文件系统上的生产力非常差，因为安装一个包含某种 ML 训练框架所需各种包的 conda 环境，要花多达 30 分钟；我们还注意到 `python -c "import torch"` 可能要花 20 多秒。这比一块高速的本地基于 NVME 的文件系统所能达到的水平慢了大约 5-10 倍。这显然是糟糕的。于是我用 `time` 设计了一个感知测试，给这些常见操作计时。这样我们就能快速判断：我们正在考虑换用的候选共享文件系统方案是否明显更好。我们不想要快 2 倍的方案，我们想要的是好 10 倍的方案，因为让一位昂贵的开发者像俗语说的"等油漆干"那样干等着，对企业来说不是好事。

下面是我们用过的"穷人版"基准测试，只是一个示例。想想你的开发者的工作流程，你肯定能很快发现哪里慢，并设计出最贴合你自己需求的测试。

注意：为了有一个可比较的基线，请在一块近期生产的本地 NVME 上跑这些计时测试。这样你就知道上限在哪里，但要注意，很多共享文件系统达不到这个水平。

第 1 步。如果你要测试的共享文件系统上还没有 conda，先把 conda 安装到那里。

```bash
export target_partition_path=/mnt/weka  # 改我！！！
mkdir -p $target_partition_path/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $target_partition_path/miniconda3/miniconda.sh
bash $target_partition_path/miniconda3/miniconda.sh -b -u -p $target_partition_path/miniconda3
rm -rf $target_partition_path/miniconda3/miniconda.sh
$target_partition_path/miniconda3/bin/conda init bash
bash
```
注意：

- 如果你不在 x86 平台上，请调整 `target_partition_path` 和 miniconda 下载链接。
- 最后我们启动一个新的 `bash` shell，让 conda 的配置生效。如果你不是 `bash` 用户，可能还需要进一步调整——我相信你知道该怎么做。

第 2 步。测量 conda 的安装耗时（写测试）

给创建一个新的 conda 环境计时：
```bash
time conda create -y -n install-test python=3.12
```

```
real    0m22.790s
user    0m12.911s
sys     0m4.941s
```

给安装一些重量级的 pip 包计时：
```bash
conda deactivate
conda activate install-test
time pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

```
real    0m50.596s
user    0m28.060s
sys     0m4.398s
```

请注意，这个测试有些失真，因为它把包的下载也计了进去，而下载取决于你的下行网络速度，可能极快也可能极慢，从而影响结果。但下载的包一旦被缓存（对 conda 来说还会被解包），你第二次安装这些包时，基准测试就不再公平了——因为在慢速共享文件系统上，解包可能非常慢，而这正是我们想捕捉到的。

我倒不太担心这一点，因为通常文件系统非常慢的时候，即使下载也很慢，你还是能看出它非常慢——只要盯着进度，你就能看出来。

如果你确实想让这个基准测试更精确，可以保留预先下载好的 conda 包，只删除它们解包出来的目录：

```bash
find $target_partition_path/miniconda3/pkgs -mindepth 1 -type d -exec rm -rf {} +
```

对 `pip` 来说，它不解包，只是缓存它下载过的 wheel 包，所以第二次运行 `time pip install` 基准测试肯定能更精确（第一次是下载、缓存并安装，第二次是从缓存安装）。所以你可以这样做：

```bash
conda create -y -n install-test python=3.12
conda activate install-test
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
conda create -y -n install-test2 python=3.12
conda activate install-test2
time pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
可以看到，这里我们只对第二次安装 pip 包计时。

如果你想要一个不掺入下载的写测试，可以把你已有的某个包复制到被测文件系统上，然后给这个操作计时。下面是把 `torch` 包——1.67GiB，分散在 13019 个文件中——复制到一个本地 NVMe 盘和一个共享 Lustre 挂载点的结果：

```bash
$ export SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')

$ time cp -a $SITE_PACKAGES/torch /tmp/torch-copy

real    0m2.026s
user    0m0.073s
sys     0m1.133s

$ time cp -a $SITE_PACKAGES/torch $target_partition_path/torch-copy

real    2m35.880s
user    0m0.266s
sys     0m9.262s
```

这慢了 77 倍，而几分钟前的另一次重复运行也花了 2m30.0s，所以这不是偶然。两次运行的 `user` 和 `sys` 几乎没有变化，因此这段时间几乎全都花在等待文件系统上——那 13019 个文件大多是小的，每一个都是一次必须由元数据服务器应答的创建操作。


第 3 步。清除内存和文件系统缓存之后测量加载耗时（读测试）

```bash
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
time python -c "import torch"
```

可以看到，在测量之前，我们必须先让操作系统清除它的内存和文件系统缓存。

如果你没有 `sudo` 权限，可以跳过涉及 `sudo` 的命令；有时系统也会配置成不需要 `sudo` 就能运行这些操作。如果你无法执行同步和清除文件系统缓存的操作，得到的结果就不正确，因为基准测试测量的将只是加载已被缓存的文件系统对象的时间。要解决这个问题，要么请系统管理员帮你执行，要么干脆第二天早上再来——届时希望你的文件系统已经缓存了别的东西、把 Python 包换出了缓存——然后再重复运行那条 Python 单行命令，并希望那些文件已经不在缓存里了。


来看看怎样观察缓存的效果：
```bash
$ sudo sync
$ echo 3 | sudo tee /proc/sys/vm/drop_caches
$ time python -c "import torch"

real    0m2.438s
user    0m8.115s
sys     0m0.434s

$ time python -c "import torch"

real    0m1.333s
user    0m8.144s
sys     0m0.214s

$ sudo sync
$ echo 3 | sudo tee /proc/sys/vm/drop_caches
$ time python -c "import torch"

real    0m2.357s
user    0m8.177s
sys     0m0.368s
```

你可以看到，第一次没有缓存，耗时更长；我第二次运行时更快，因为一切都已被缓存。然后我又让系统清除内存和文件系统缓存，可以看到它又变慢了。

了解这些秒数花在哪里是有帮助的。一次导入过程中会发生两类不同的 IO：对 `torch/lib` 中的共享对象文件（shared objects）做少量大型顺序读取，以及几千次微小的操作——每个模块一次 open，外加 Python 为每个模块探测候选路径时的那些失败查找。[`strace`](../debug/pytorch.md#strace) 统计的是第二类：

```bash
$ strace -cf -e trace=openat python -c "import torch"
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
100.00    0.005894           2      2339       922 openat
------ ----------- ----------- --------- --------- ----------------
100.00    0.005894           2      2339       922 total
```

这 2339 次 open——其中 922 次失败，因为 Python 会为每个模块探测多个候选路径——是由所安装的包及其版本决定的。而每一次 open 要花多久，则是由文件系统决定的，这个差距并不小。在本地 NVMe（同一节点）和它的共享 Lustre 挂载点上打开并关闭几千个小文件的测量结果：

| 单个小文件的一次打开 | 本地 NVMe | Lustre |
| :--------------------- | ---------: | -----: |
| 成功 |       11us |  2.3ms |
| 以 `ENOENT` 失败 |      2.4us |  1.1ms |

把这些数字套用到 `import torch` 实际执行的操作组合上——1417 次成功的 open 和 922 次失败的 open——仅 open 一项就从本地盘上的约 0.02 秒涨到 Lustre 上的约 4.3 秒。当一个共享文件系统每次 open 的延迟处于这个范围时，1.3 秒的导入就会变成上面案例中报告的 20 秒。把所有内容都预先读过一遍，再重复测量 Lustre，结果同样没有任何变化：因为付出代价的是一次元数据往返，而页缓存（page cache）对此帮不上忙。

open 是一个 [IOPS](#metrics) 问题，而对 `torch/lib` 中共享对象文件的大量顺序读取是一个吞吐量问题，一个文件系统完全可能擅长其一却拙于另一个。

我觉得在写测试中同样清除内存和文件系统缓存可能是个好主意，因为即便在那里，缓存也会让基准测试显得比真实情况更快，而真实世界里一个新包是第一次被安装。

还有一件事，即使清除缓存后的读测试也捕捉不到。安装之后的第一次导入，会把每个被触及的 `.py` 文件编译成字节码，并把 `.pyc` 写入 `__pycache__`；之后的每一次导入都会读取 `.pyc` 并跳过编译。清除缓存并不能撤销这一点，因为 `.pyc` 文件还在磁盘上——所以安装后真正的第一次运行，会比上面任何一个数字都慢，而随手运行一次 `python -c "import torch"` 就足以留下这种已编译的状态。

还有一次，我注意到 `git status` 要花好几秒。我在用 [bash-git-prompt](https://github.com/magicmonty/bash-git-prompt)，当处于一个 git 仓库克隆内时，它每次返回提示符之前都会运行 `git status`，结果终端变得极其迟缓，很难工作。于是我对 `git status` 做了基准测试：

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
time git status
```
在这个慢速文件系统上它要花 3.7 秒，必须得修（在本地 SSD 上只要 0.02 秒）。好处是这个真实的感知基准测试很容易交给系统管理员，他们能立刻复现问题，然后着手修复，同时把这个基准测试当作参照反复使用。

又有一次，我注意到 `pytest` 启动慢得仿佛没有尽头，于是我测量了它的用例收集（collection），果然非常慢：

```bash
time pytest --disable-warnings --collect-only -q
```

所以现在你有大量示例可以选择，我相信你一定能找到容易可靠复现的、属于你自己的用例，把它们作为参照点，判断什么体验好、什么体验不好、哪些需要修复。




### 其他工具 {#other-tools}

要想对共享/并行文件系统做出比上文[临时计时做法](#usability-perception-io-benchmarks)更严格、可复现的基准测试，有两个工具值得了解：

- [IOR](https://github.com/hpc/ior) - HPC 领域标准的并行 IO 基准测试。它通过 MPI 启动大量进程并发读/写，因此能测出分布式文件系统在真实并行负载下、跨多种后端（POSIX、MPI-IO、HDF5 等）所能维持的*聚合*顺序与随机带宽。同一个仓库还提供了 `mdtest`（2017 年并入 `ior`），它测量的是*元数据*性能——文件系统每秒能处理多少次文件/目录的创建/统计（stat）/删除操作。对包含大量小文件的工作负载（比如许多小 checkpoint 或数据集分片）来说，元数据往往才是真正的瓶颈。
- [DLIO](https://github.com/argonne-lcf/dlio_benchmark) - 一个深度学习 IO 基准测试，不需要任何加速器就能模拟 ML 训练的数据加载模式：它用等时长的 `sleep` 替代前向/反向计算，并按照 YAML 工作负载配置描述的合成数据集驱动真正的数据加载器（PyTorch `DataLoader`、`tf.data`）。一次运行分三个阶段——生成合成数据、运行基准测试、后处理生成报告。这使它成为最接近"我的存储在真实训练中会表现如何"的代理，而 IOR 报告的只是通用的带宽/IOPS 数字。



### 已发布的基准测试 {#published-benchmarks}

下面是一些公开发布的 IO 基准测试：

- [MLPerf via MLCommons](https://mlcommons.org/) 发布多种硬件基准测试，衡量训练、推理、存储和其他任务的性能。这里相关的是[存储结果](https://mlcommons.org/benchmarks/storage/)——那个页面总是显示当前轮次，而轮次一年会推进好几次。不过我觉得这些结果非常难看懂——列太多，用户完全没有任何控制权，而且每个测试用的参数都不同——那要怎么比较呢。




## 明明可以轻松清理，何必花钱买更多存储 {#why-pay-for-more-storage-when-you-can-easily-clean-it-up-instead}

与几家存储服务商交流后我了解到，很多公司根本懒得清理，只是一个劲地越买越多。如果你不想做这样的公司，希望把环境保持整洁，那么在下面几节中，我会分享如何轻松清理我们这些身处 Python/PyTorch 生态中的人常用的各种缓存（其中不少做法也适用于其他生态）。
### HuggingFace Hub 缓存 {#huggingface-hub-caches}

非常流行的 HuggingFace Hub 让下载模型和数据集并在本地缓存它们变得极其简单。你可能没有意识到的是，每当模型或数据集发布新的修订版本（revision）时，旧的修订版本仍然会留在你的磁盘上——所以随着时间推移，你很可能积累了大量无用的负担。

缓存文件通常位于 `~/.cache/huggingface`，但要把它们放到别处，最不容易出错的方法是在导入 Hugging Face 库之前设置 [`HF_HOME` 环境变量](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)：

```bash
export HF_HOME=/some/path/huggingface
mkdir -p "$HF_HOME"
```

在较旧的环境中，你可能仍会遇到已弃用的 `HUGGINGFACE_HUB_CACHE` 以及诸如 `TRANSFORMERS_CACHE` 这类特定库的变量。

另一种不需要环境变量的解决方案，是把你的缓存软链接（symlink）到另一个分区。你可以对你所有的缓存这样做：
```bash
mkdir -p /some/path
mv -nT ~/.cache /some/path/.cache
ln -s /some/path/.cache ~/.cache
```

或者只移动默认的 Hugging Face 缓存，并在原位置留下一个软链接。这个脚本会直接停止，而不是与已存在的目标合并，也不是替换已存在的软链接：
```bash
#!/usr/bin/env bash
set -euo pipefail

source_dir="$HOME/.cache/huggingface"
target_dir="/some/path/huggingface"

mkdir -p "$(dirname "$source_dir")" "$(dirname "$target_dir")"

if [ -e "$target_dir" ] || [ -L "$target_dir" ]; then
    printf 'Target already exists: %s\n' "$target_dir" >&2
    exit 1
fi

if [ -L "$source_dir" ]; then
    printf 'Source is already a symlink: %s\n' "$source_dir" >&2
    exit 1
elif [ -e "$source_dir" ]; then
    mv "$source_dir" "$target_dir"
else
    mkdir -p "$target_dir"
fi

ln -s "$target_dir" "$source_dir"
```

源目录不存在的分支会在建立链接前创建一个空的目标目录。如果目标里已经包含了应当复用的缓存，请先自行检查并手动创建软链接，而不是让这个脚本来合并目录。

现在你已经知道缓存在哪里了，当然，你可以时不时把整个缓存删掉，但如果这些是巨大的模型和数据集，尤其后者还做过一些预处理——你真的不会想一遍又一遍地重复那些耗时的任务。所以我会教你如何使用 HuggingFace 提供的专门工具来做清理。

HF hub 上修订版本的工作机制，是让 `main` 指向文件的最新修订版本，同时把旧的修订版本保留下来，以防有人出于某种原因需要使用旧版本。你极大概率永远只想要最新的修订版本，因此以下是删除所有旧修订版本、只保留 `main` 的方法：

```bash
pip install -U "huggingface_hub"
hf cache prune
```

`hf cache prune` 会删除所有已经没有任何分支、标签或拉取请求（pull-request）引用指向的修订版本——这正是你想清掉的旧游离（detached）修订版本——同时还会删除被中断的下载遗留下来的任何 `.incomplete` 数据块（blob）。先用 `--dry-run` 看看会清理多少内容而实际不删除任何东西：

```bash
$ hf cache prune --dry-run
About to delete 30 unreferenced revision(s) and 9 incomplete download(s) (180.9G total).
  - dataset/fan-shu/instruct2thinking:
      4c8e57ed5f06425ce0d743cf168b906646f0a338 [(detached)] 609.7M
      ce82e962df666fc85657a3a871443009ec825b5f [(detached)] 259.7M
      d5a3ba55f8ac010ee0f1ac86314b49475a1e300a [(detached)] 226.2M
  - dataset/fan-shu/swe-instruct-trajectories-empty-think-inserted:
      84d55e1f36605fe990feb61a1a705032f0620015 [(detached)] 12.1G
  - model/Qwen/Qwen3.5-27B:
      b7ca741b86de18df552fd2cc952861e04621a4bd [(detached)] 55.6G
  - model/zai-org/GLM-5.2:
      f2263102df303b2faa54a6861a29d1770ce846c0 [(detached)] 1.5T
✓ Dry run: no files were deleted.
```

总数是共享数据块只计算一次之后的可回收空间；上面按修订版本列出的大小加起来可能远多于此。不带 `--dry-run` 时，它会在删除前征求确认；如果你要在 cron 作业里运行它，加上 `--yes` 可以跳过确认提示。

在删除任何东西之前，如果你想弄清楚空间都去哪儿了，`hf cache ls` 会给出每个仓库的总量，`hf cache ls --revisions` 则按修订版本细分。它支持过滤器，过滤器能理解人类可读的大小和时长，所以你可以去找那些大的和被遗忘的东西：

```bash
hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
```

而要删除整个模型或数据集、而不只是它们陈旧的修订版本，`hf cache rm` 接受仓库 id 和修订版本哈希：

```bash
hf cache rm model/bert-base-cased dataset/glue
```

它可以与 `hf cache ls -q`（只输出标识符，每行一个）配合使用，按过滤器删除：

```bash
hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
```

如果你的缓存不在默认位置，上面这些命令中的每一个都接受 `--cache-dir PATH`。

另外你会发现，HF `datasets` 有一个 `~/.cache/huggingface/datasets/downloads` 目录，其中通常会有大量从数据集下载及其预处理中遗留下来的东西，包括各种锁文件。在一个环境里，我在那里发现了实实在在几百万个文件。下面是我清理它们的方式：

```bash
sudo find ~/.cache/huggingface/datasets/downloads -type f -mtime +3 -exec rm {} \+
sudo find ~/.cache/huggingface/datasets/downloads -type d -empty -delete
```

第一条命令会保留存在时间不足 3 天的文件，以防有人正在下载或处理东西，我们不想突然抽走他们脚下的地毯。

和往常一样，如果你把缓存放在了别处，可能需要调整这些路径。

注意：如果你的团队使用 `HF_HOME` 来共享 HF hub 上的模型/数据集等——`$HF_HOME/token` 也会被一起共享，只要使用的都是非受限模型（ungated models），这样没问题。但如果你想访问受限模型（gated models，即需要申请授权才能访问的模型），就可能遇到麻烦。因此你很可能不希望共享访问令牌。你可以通过加上类似下面的设置来解决这个问题：

```bash
export HF_TOKEN_PATH=~/.cache/hf_hub_token
```
（然后把它放进 `~/.bashrc`，让它始终生效）

现在让每个用户运行一次：
```
hf auth login
```
这会提示他们添加自己的访问令牌，令牌从 https://huggingface.co/settings/tokens 获取——令牌会被保存到 `~/.cache/hf_hub_token` 下。

这样，你团队的每个成员都会有各自独立的令牌，那些为他们的 HF hub 用户批准的受限模型也就变得可以访问了。


### Python 包管理器清理 {#python-package-manager-cleanups}

conda 和 pip 会随着时间推移在你的系统上堆积越来越多的文件。conda 是最糟糕的，因为它保留解包后的文件，这些文件会消耗数量惊人的 inode，并让备份和扫描变慢。pip 至少只缓存 wheel 包（打包文件）。

所以你可以放心地删除这些目录：

```bash
rm -rf ~/.cache/pip
rm -rf ~/anaconda3/pkgs/
```

如果你的 conda 安装在别处，一定要修改最后一条命令。


### 在团队环境中共享缓存 {#share-caches-in-group-environments}

如果有超过 2 个人在同一台系统上工作，你最好不要让每个人都各自维护自己的 `pip`、`conda`、HF 模型、数据集以及可能的其他东西的缓存。让每个用户的配置指向共享缓存非常容易。

例如，假设你按如下方式在 `/data/cache` 下创建 `pip` 和 `conda` 缓存：

```bash
mkdir /data/cache/conda
mkdir /data/cache/pip
chmod a+rwx /data/cache/conda
chmod a+rwx /data/cache/pip
```

现在你只需要把每个用户的本地缓存软链接到这个共享缓存：
```bash
mkdir -p ~/.cache

rm -rf ~/.cache/pip
ln -s /data/cache/pip ~/.cache/pip

rm -rf ~/.conda/pkgs
ln -s /data/cache/conda/pkgs ~/.conda/pkgs
```
注意，我们把已有的缓存直接删掉了，但你也可以把它们移动到共享缓存里——怎么管用就怎么做，反正你之后也需要定期清理这些缓存。

这样，当 `pip` 或 `conda` 试图访问用户缓存时，就会被重定向到共享缓存。如果团队有 20 个人，文件数量就少 20 倍——这一点非常重要，因为 conda 的包文件是解包状态的，会占用磁盘上大量的 inode。

所以这种做法唯一的问题就是文件权限。如果用户 A 安装了一些包，用户 B 可能无法读写它们。

如果这是一个没有恶意用户的独立集群，你可以直接请每个人在 `~/.bashrc` 中使用 `umask 000`，甚至可以通过 `/etc/profile` 或 `/etc/bash.bashrc` 把该设置配置到整个系统；如果 `bash` 不是你选择的 shell，还有其他各种 shell 配置文件可用。

一旦运行了 `umask 000`，大多数文件在创建时都会带有读写权限，从而所有用户都可以读写彼此的文件。

当然，如果你使用的是某种 HPC，许多互不相关的团队共用同一个集群，这种做法就行不通了。这时你可以使用用户组，而不是让所有文件对所有用户可读写，也许还会预置 `setgid` 位，或者使用 ACL。在这类环境中总是有系统管理员的，所以你可以问他们如何为你的团队设置共享缓存，他们会知道该怎么做。

另外，最近这些应用中的一些加入了清理工具，例如 `conda` 和 `pip`：
```bash
conda clean --all -f -y
pip cache purge
```

### 磁盘用量总览 {#general-disk-usage}

当然，迟早你的分区会变得越来越大，你可能会想弄清楚数据都在往哪里漏。通常你需要找出贡献了最多数据占用的用户，请他们做一些清理。

例如，要找出哪些用户占用了最多的磁盘，运行：
```bash
sudo du -ahd1 /home/* | sort -rh
```
它会按占用最严重的排在最前面来排序这些数据。如果你想帮他们一把，可以进入他们的目录，再深入一层分析数据：

```bash
sudo du -ahd1 /home/*/* | sort -rh
```
或者针对特定用户 `foo`：
```bash
sudo du -ahd1 /home/foo/* | sort -rh
```

你也可以设置磁盘使用配额，但这通常不太管用，因为取决于贵公司的工作流程，有些用户天然需要比其他用户生成多得多的数据，不应该因此受到的惩罚是做不了自己的工作、让作业崩溃——作业可能已经跑了好几个小时，所有工作都会丢失——所以归根结底，公司要为损失的时间买单。

让用户意识到自己占用了太多磁盘空间，可能是一件非常困难的事情。

### 分区的 inode 上限 {#partition-inodes-limit}

另外要小心 inode 的使用量。在一些 HPC 的共享分区上，我不止一次遇到过这种情况：作业崩溃并不是因为磁盘空间用完了，而是因为作业用掉了最后的 inode，导致整个事情崩溃了。

要查看 inode 使用情况，使用 `df -i`：
```bash
$ /bin/df -hi
Filesystem     Inodes IUsed IFree IUse% Mounted on
tmpfs             16M  1.9K   16M    1% /run
/dev/sda1         59M  4.1M   55M    7% /
```
`-h` 会把巨大的数字格式化为人类可读的字符串。

所以从这里你可以看到，`/` 分区使用了 inode 总数的 7%。

取决于文件系统类型，有些情况下可以增加更多 inode，有些情况则不行。

因此，作为磁盘空间监控的一部分，你还需要把 inode 使用量作为一项关键资源来监控。



### 计算节点上的 `/tmp` {#tmp-on-compute-nodes}

通常计算节点会用 `/tmp/` 存放临时文件。问题在于，在大多数配置下，`/tmp` 位于每个节点很小的 `/` 文件系统上（常常小于 100GB），而 `/tmp/` 只在重启时才会重置，因此在 SLURM 作业之间它不会被清理，这会导致 `/tmp` 的空间耗尽。于是当你尝试运行某个比如要解包一个文件的东西时，你很可能会遇到：

```bash
OSError: [Errno 28] No space left on device
```

解决方案是在你的 SLURM 启动脚本中设置。
```bash
export TMPDIR=/scratch
```

这样，slurm 作业就会使用大得多的 `/scratch` 而不是 `/tmp`，因此有充足的临时空间可以写入。

脚注：虽然 `/scratch` 很常见——挂载的本地 SSD 磁盘的挂载点可能叫任何名字，例如 `/localssd`——但在某个计算节点上运行 `df`，应该很容易看出正确的路径。

你也可以安排 SLURM 配置在作业结束时自动清理这类目录。


### 如何找出检查点占用大量磁盘空间的用户 {#how-to-find-users-whose-checkpoints-consume-a-lot-of-disk-space}

你有没有遇到过这种问题：团队在训练模型，你却不得不不断购买更多存储，因为巨大的模型检查点没有被足够快地转储（offload）到存储桶（bucket storage）？

这里有一个一行命令，它会递归分析你选择的某个路径，找出所有检查点，累加它们的大小，并按占用最大的用户排序打印总计，这样你就可以提醒他们收拾一下自己的烂摊子 :) 只需把 `/mypath` 改成实际路径：

```bash
find /mypath/ -type f -regextype posix-egrep -regex ".*\.(pt|pth|ckpt|safetensors)$" | \
perl -nle 'chomp; ($uid,$size)=(stat($_))[4,7]; $x{$uid}+=$size;
END { map { printf qq[%-10s: %7.1fTiB\n], (getpwuid($_))[0], $x{$_}/2**40 }
sort { $x{$b} <=> $x{$a} } keys %x }'
```

这会生成类似这样的输出：
```
user_a   :     2.5TiB
user_c   :     1.6TiB
user_b   :     1.2TiB
```

当然，你可以修改正则表达式以匹配其他模式，或者干脆把它整个去掉，从而统计所有文件：

```bash
find /mypath/ -type f | \
perl -nle 'chomp; ($uid,$size)=(stat($_))[4,7]; $x{$uid}+=$size;
END { map { printf qq[%-10s: %7.1fTiB\n], (getpwuid($_))[0], $x{$_}/2**40 }
sort { $x{$b} <=> $x{$a} } keys %x }'
```

如果你想高效地排除某些子目录：

```bash
find /mypath/ -regextype posix-egrep \
-type d -regex "/mypath/(exclude_a|exclude_b|exclude_c)" -prune -o \
-type f -regex ".*\.(pt|pth|ckpt|safetensors)$" -print | \
perl -nle 'chomp; ($uid,$size)=(stat($_))[4,7]; $x{$uid}+=$size;
END { map { printf qq[%-10s: %7.1fTiB\n], (getpwuid($_))[0], $x{$_}/2**40 }
sort { $x{$b} <=> $x{$a} } keys %x }'
```

提示：第二行告诉 `find` 跳过匹配 `/mypath/(exclude_a|exclude_b|exclude_c)` 正则表达式的文件夹。有两个细节：第一，正则表达式必须与文件夹路径完全匹配，因为如果你在末尾追加一个 `/.*`，`find` 会进入该文件夹本身、只剪除（prune）它的子文件夹——所以直接放在该文件夹里的文件仍然会被统计进去。第二，`-print` 是必需的，因为 `-prune` 是一个动作，没有第二个动作时，`find` 会把被剪除的文件夹也打印出来，这些文件夹随后会被 `stat`，并被加到某个用户的总计里。请按需调整以适配你的使用场景。


### 如何自动删除旧检查点 {#how-to-automatically-delete-old-checkpoints}

承接[找出检查点占用磁盘空间最多的用户](#how-to-find-users-whose-checkpoints-consume-a-lot-of-disk-space)，这里说明如何自动删除旧的检查点（例如超过 30 天的那些）。

首先设法确认这些候选文件确实适合删除：

```bash
find /mypath/ -regextype posix-egrep -regex ".*\.(pt|pth|ckpt|safetensors)$" -mtime +30
```

当你觉得可以安全删除了，再添加 `rm`
```bash
find /mypath/ -regextype posix-egrep -regex ".*\.(pt|pth|ckpt|safetensors)$" -mtime +30 -exec rm {} +
```

## 资源 {#resources}

### du 的替代工具 {#du-alternatives}

`du` 是一个强大的 Unix 工具，但它可能很慢，而且通常需要对结果做额外的后处理排序，才能让结果有用。例如，我经常使用 `du -ahd1 | sort -rh` 这个组合命令来获取单层目录的磁盘用量并按大小排序。

- [dust](https://github.com/bootandy/dust) - 用 Rust 编写，运行比 `du` 更快，并且内置排序。我觉得不带任何参数的输出读起来令人困惑，因为它把许多层混在一起并分别排序，但别人可能觉得没问题。`dust -F` 和 `dust -D` 看起来非常有用，而且格式排版得很好。
- [baobab](https://github.com/GNOME/baobab) - 分析分区的速度相当快，并提供一个 UI，可按子文件夹大小浏览结果，还便于复制路径，在需要删除大量旧的大型数据集和检查点文件夹时很有用。我认为这个工具只在桌面上有用，因为它需要 GNOME 环境。
