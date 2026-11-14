# 网络调试 {#network-debug}

很多时候，你并不需要成为网络工程师也能弄清楚网络问题。阅读下面的笔记，一些常见问题就能得到解决。



## 术语表 {#glossary}

- Bonding（网卡绑定）：把多张 NIC 一起使用，以获得更快的速度或作为备份
- HCA：主机通道适配器（Host Channel Adapter）——InfiniBand 对网络适配器的称呼
- IB：InfiniBand（最初由 Mellanox 开发，后被 NVIDIA 收购）
- NIC：网络接口卡（Network Interface Card）
- OOB：带外（Out-of-Band），通常是一张速度较慢的以太网 NIC
- RDMA：远程直接内存访问（Remote Direct Memory Access）


## 如何诊断 NCCL 多 GPU 与多节点连接问题 {#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues}

另见[调试多节点训练](../../debug/pytorch.md#debugging-multi-node-training)，介绍如何在 PyTorch 层面调试多节点问题。

本节显然不面面俱到，只是打算覆盖一些我在实际中经常遇到的最常见的配置问题。对于更复杂的问题，请研究 [NCCL 仓库的 Issues](https://github.com/NVIDIA/nccl/issues)，如果找不到与你情况匹配的 Issue，就新建一个。NCCL 文档也包含一个简短的[故障排查小节](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)，但通常阅读 [Issues](https://github.com/NVIDIA/nccl/issues) 能学到更多。

做网络诊断时，我不建议使用一个完整的应用程序——它可能启动很久，而且还会牵涉无关的问题；我建议使用这个专门开发的、用于设计验证的测试脚本：[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)。

首先，在设置了下面的内容之后再运行基于 nccl 的程序：

```bash
export NCCL_DEBUG=INFO
```
这会打印大量关于 NCCL 设置及其网络流量的调试信息。

例如，如果你使用前面提到的那个调试脚本，在一台带有 8 块 GPU 的单节点上，你可以这样做：

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

要在多个节点上启动它，你要么使用某种编排软件，比如 SLURM 或 Kubernetes，要么在每个节点上手动启动（`pdsh` 会帮上大忙）——详见 [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py) 内部的说明。不过，为了理解整个机制是如何运作的，我建议先只在 1 个节点上开始，然后再推进到 2 个，之后再到更多节点。

现在，检查程序的输出，寻找以如下内容开头的行：
```
NCCL INFO NET/
```
然后检查它正在使用哪个协议以及哪些接口。

例如，下面这段输出：
```
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

告诉我们使用的是 [nccl-fastsocket](https://github.com/google/nccl-fastsocket) 传输层插件，它发现了 8 个 `ibs*` 网络接口（NIC 卡）。如果你用的是 Google Cloud，这是正确的，你的 NCCL 很可能也就配置正确。但如果你使用的是 InfiniBand (IB)，却得到了上面的输出，那么你很可能只会得到非常低的节点间速度，因为这意味着你激活了错误的插件。

在使用 IB 的情况下，你应当看到的是 `NET/IB` 及其 IB 接口：
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

在这里你可以看到，IB 使用 8 个 `mlx5_*` 接口来进行集合通信，还有一个 OOB，即带外（Out-Of-Band），它用于引导建立连接，通常使用一张速度较慢的以太网 NIC（有时[会把多张 NIC 绑定成一张](https://wiki.linuxfoundation.org/networking/bonding)——如果你正好奇接口名里的 `bond` 是什么意思，就是指这个）。

要知道你的节点有哪些 TCP/IP 接口，可以在其中一台节点上运行 `ifconfig`（通常所有同型号节点都会有相同的接口名，但并不总是如此）。

如果你的集合通信网络是 IB，那么你要运行的不是 `ifconfig`，而是 `ibstat`。上面最后一个 `NCCL INFO NET` 示例会对应如下输出：

```bash
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

这些命令中的哪些能真正给出回答，取决于网络架构（fabric），所以了解每个命令问的是什么会有帮助。`ibstat` 与 `ibnetdiscover`、`iblinkinfo`、`ibhosts`、`ibswitches` 和 `perfquery` 一样，都是向子网管理器（subnet manager）查询——子网管理器是覆盖整个网络架构的服务，在 InfiniBand 上负责分配地址并回答拓扑问题。它们只对真正的 InfiniBand 网络架构给出报告，上面的示例就是这种情况。而 `ibv_devices`、`ibv_devinfo` 和 `ibstatus` 询问的是本地适配器，所以只要是被 RDMA 协议栈枚举出来的网络架构，它们都能回答；直接用 `cat /sys/class/infiniband/*/ports/1/rate` 读取端口也是如此。

所以要先确定网络架构，使用第二组中的某个命令。在 InfiniBand 上，端口的 `link_layer` 是 `InfiniBand`，并且子网管理器工具可用。在 RoCE 上——它把 RDMA 承载在以太网之上——`link_layer` 是 `Ethernet`。在 EFA 上没有子网管理器：`ibv_devinfo` 报告 `transport: unspecified` 和 `link_layer: Unspecified`，`ibstatus` 报告每个端口都以各自的速率处于 `ACTIVE`，而子网管理器工具会失败——`ibnetdiscover` 报 `Can't open SMI UMAD port`，`ibstat -l` 则什么都不输出。这种毫无动静的空输出正是陷阱所在，因为它看起来像是"这个节点没有 RDMA 设备"；在本章为实测而使用的一台 2026 年 9 月的 AWS 节点上，它什么都没打印，而该节点其实有 16 块适配器，每一块都在 `200 Gb/sec (4X HDR)` 速率下处于 `ACTIVE`。

除了高速的节点间连接 NIC 之外，你很可能还有一张低速的管理用以太网 NIC（甚至好几张），它的存在是为了让你能配置节点、使用共享文件系统、访问互联网，所以几乎可以肯定，`ifconfig` 的输出还会包含额外的 NIC。此外你还可能有 docker 网络接口、`lo` 回环接口以及其他一些接口。例如，在我的台式机上，我可能得到如下输出：

```bash
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

我之所以提到这一切，是因为关键部分在于确保 NCCL 在其 `Using` 调试行中只报告正确的接口。如果像 `docker0`、`lo` 或 `eth0` 这样的接口最终也被报告出来，例如：

```
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

而你有更快的网络接口可用，那么这多半不是你想要的。当然，在某些情况下，以太网 NIC 是你仅有的选择，这种情况下上面的输出也没什么问题——只是会非常慢。

有时候，如果最终用上了错误的接口，应用程序可能直接挂起。

如果你拥有的既有全部正确的接口，又夹杂着一些不正确的接口，NCCL 也许能工作，但速度会更慢。

如果这是云环境，通常你的云服务商应该会给出让配置正确起来的说明。如果他们没给，那么至少要问清楚：为了配置 NCCL，你需要使用哪些网络接口。

尽管 NCCL 会尽力自动发现它该使用哪些接口，但如果它没能正确地做到这一点，你就可以帮它一把，告诉它该用或不该用哪些接口：

- 在不使用 InfiniBand 时，`NCCL_SOCKET_IFNAME` 可用来指定要包含或排除哪些 `ifconfig` 接口。下面是一些例子：

```bash
export NCCL_SOCKET_IFNAME=eth         # 所有以 eth 开头的接口
export NCCL_SOCKET_IFNAME==eth0       # 仅 eth0
export NCCL_SOCKET_IFNAME==eth0,eth1  # 仅接口 eth0 和 eth1
export NCCL_SOCKET_IFNAME=^docker     # 不使用任何以 docker 开头的接口
export NCCL_SOCKET_IFNAME=^=docker0   # 不使用接口 docker0
```
完整文档见[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)。

- 当使用 IB RDMA（IB Verbs 接口）时，不要用 `NCCL_SOCKET_IFNAME`，而要改用环境变量 `NCCL_IB_HCA`，它为集合通信选择接口。示例：

```bash
export NCCL_IB_HCA=mlx5                # 名称以 mlx5 开头的每张卡的全部端口
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1  # 仅 mlx5_0 和 mlx5_1 的端口 1
export NCCL_IB_HCA=^=mlx5_1,mlx5_4     # 不使用 mlx5_1 和 mlx5_4
```
完整文档见[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)。

例如，在使用 IB 时，常常会有诸如 `mlx5_bond_0` 这样的额外接口，你并不希望它们被纳入 NCCL 通信。比如，下面这份报告就说明错误的 `[8]mlx5_bond_0:1/RoCE` 接口被包含进来了，而这几乎肯定会导致低带宽：
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```
此时你可以用下面的方式把它排除掉：
```bash
export NCCL_IB_HCA=^mlx5_bond_0:1
```
或者，你也可以显式列出你想要的接口，例如：
```bash
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

如前所述，如果网络架构是 InfiniBand，在任一节点上运行 `ibstat` 就会列出可用的接口；在任何其他 RDMA 网络架构上，则使用 `ibv_devices` 或 `ibv_devinfo`。

由于 NCCL 会尝试自动选择最佳的网络接口，你只有在 NCCL 无法工作或速度很慢时才需要做上面的事情。正常情况下，NCCL 应当开箱即用，用户不需要做任何特别的操作。

另外，取决于使用的是哪家云，云服务商很可能会给你一大堆需要设置的环境变量。如果你把其中一些设置错了，NCCL 可能会运行缓慢，或者完全无法工作。

用户还会遇到另一个典型问题：试图把在云 A 上已经能正常工作的 NCCL 配置原样复用到云 B 上。通常这些东西并不通用，你必须小心地清除之前设置过的所有环境变量，再针对新的云重新正确地设置一遍。即使你用的是同一家云、但不同类型的实例，也很可能出现这个问题，因为有些网络配置是针对特定实例的，换个环境就行不通。

一旦你认为已经把 NCCL 配置正确了，下一件事就是对你的连接做基准测试，确认它达到了标称的速度（好吧，大约是其 80%）。请转到[基准测试章节](../benchmarks/README.md)。


## NCCL 与 docker 容器 {#nccl-with-docker-containers}

* 通过向 docker `run` 添加这些额外参数来给出足够的资源：`--shm-size=1g --ulimit memlock=-1`（[更多细节](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#sharing-data)）
* 特权访问：有时你需要在 docker `run` 的参数里加上 `--privileged`。
* 让 docker 镜像包含正确的软件包，例如，如果使用 IB，你至少需要安装 `libibverbs1 librdmacm1`



## 如何检查是否支持 P2P {#how-to-check-if-p2p-is-supported}

有时你需要知道计算节点上的 GPU 是否支持 P2P 访问（Peer2Peer，点对点）。禁用 P2P 通常会导致节点内部连接变慢。

可以看到，在这台特定的 8x NVIDIA H100 节点上，P2P 是受支持的：

```bash
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 GPU1   OK      X       OK      OK      OK      OK      OK      OK
 GPU2   OK      OK      X       OK      OK      OK      OK      OK
 GPU3   OK      OK      OK      X       OK      OK      OK      OK
 GPU4   OK      OK      OK      OK      X       OK      OK      OK
 GPU5   OK      OK      OK      OK      OK      X       OK      OK
 GPU6   OK      OK      OK      OK      OK      OK      X       OK
 GPU7   OK      OK      OK      OK      OK      OK      OK      X

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
```

另一方面，在这台特定的 2x NVIDIA L4 上，P2P 不受支持：
```bash
$ nvidia-smi topo -p2p r
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

正如从上面的 Legend（图例）中可以看到的那样，`CNS` 表示 "Chipset is not supported"（芯片组不支持）。

如果你用的是高端数据中心 GPU，这种情况几乎不会发生。不过，有些低端数据中心 GPU 可能不支持 P2P，就像上面 L4 的例子那样。

对于消费级 GPU，你的 GPU 不受支持的原因可能有很多种，通常是启用了 IOMMU 和/或 ACS 特性。另一些时候则只是驱动版本的问题。如果你多花些时间搜索，可能会发现有人在改造驱动，让本不支持 P2P 的 GPU 支持了 P2P，比如下面这个 [4090 P2P 支持仓库](https://github.com/tinygrad/open-gpu-kernel-modules)。

要检查 PCI 访问控制服务（Access Control Services，ACS）是否启用以及如何禁用它们，请遵循[这份指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#pci-access-control-services-acs)。

IOMMU 可以在 BIOS 中禁用。

你也可以用 torch 检查特定 GPU 之间的 P2P 支持——这里我们在检查 GPU 0 和 GPU 1：

```bash
python -c "import torch; print(torch.cuda.can_device_access_peer(torch.device('cuda:0'), torch.device('cuda:1')))"
```
如果不支持 P2P，上面的命令会打印 `False`。



## 如何统计 NCCL 调用次数 {#how-to-count-nccl-calls}

启用 NCCL 针对子系统——集合通信——的调试日志：
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

如果你在一个有很多节点的 slurm 环境中工作，你可能希望只在 rank 0 上这样做，如下所示：
```bash
if [[ $SLURM_PROCID == "0" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL
fi
```

假设你的日志全部发送到了 `main_log.txt`，那么你可以用下面的命令统计每种集合通信调用执行了多少次：
```bash
grep -a "NCCL INFO Broadcast" main_log.txt     | wc -l
2590
grep -a "NCCL INFO AllReduce" main_log.txt     | wc -l
5207
grep -a "NCCL INFO AllGather" main_log.txt     | wc -l
1849749
grep -a "NCCL INFO ReduceScatter" main_log.txt | wc -l
82850
```

最好先隔离出训练的某个特定阶段，因为加载与保存的模式和训练迭代的模式会非常不同。

所以我通常先切出一个迭代。例如，如果每次迭代的日志以 `iteration: ...` 开头，那么我会先执行：
```
csplit main_log.txt '/iteration: /' "{*}"
```
然后分析对应迭代的那些结果文件中的一个。默认情况下，它的名字类似 `xx02`。


## 有用的 NCCL 调试环境变量 {#useful-nccl-debug-environment-variables}

在调试与 NCCL 相关的问题（例如挂起和崩溃）时，下列环境变量最为有用。完整列表可以在[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)找到。


### `NCCL_DEBUG` {#nccl_debug}

这是调试网络问题时最常用的环境变量。

取值：
- `VERSION` - 在程序开始时打印 NCCL 版本。
- `WARN` - 每当任何一次 NCCL 调用出错时，打印一条明确的错误信息。
- `INFO` - 打印调试信息
- `TRACE` - 为每一次调用打印可回放的跟踪信息。

例如：

```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

这会转储大量与 NCCL 相关的调试信息，如果你发现有报告出来的问题，就可以拿它们去网上搜索。

而在使用 `NCCL_DEBUG` 时，`NCCL_DEBUG_FILE` 应该会非常有用，因为信息量非常大，尤其是在使用很多节点的时候。



### `NCCL_DEBUG_FILE` {#nccl_debug_file}

使用 `NCCL_DEBUG` 环境变量时，把所有 NCCL 调试日志输出重定向到一个文件。

默认值是 `stdout`。当使用很多 GPU 时，把每个进程的调试信息保存进各自独立的日志文件会非常有用，可以这样做：

```bash
NCCL_DEBUG_FILE=/path/to/nccl-log.%h.%p.txt
```

- `%h` 会被替换为主机名
- `%p` 会被替换为进程的 PID。

如果你随后需要一次分析上百个这样的文件，这里有一些有用的小技巧：

- 用 grep 查找特定的匹配项，并同时打印找到它的文件和行号：

```bash
grep -n "Init COMPLETE" nccl-log*
```

- 显示所有 nccl 日志文件的 `tail -1` 结果，并在其后附上每个文件的名称

```bash
find . -name "nccl*" -exec sh -c 'echo "$(tail -1 "$1") ($1)"' _ {} \;
```



### `NCCL_DEBUG_SUBSYS` {#nccl_debug_subsys}

`NCCL_DEBUG_SUBSYS` 与 `NCCL_DEBUG` 配合使用，用来告诉后者要显示哪些子系统。通常你不必指定这个变量，但有时帮助你的开发人员可能会要求把输出限制在某些子系统，例如：

```bash
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING
```



### `NCCL_IB_HCA` {#nccl_ib_hca}

当集合通信通过 IB Verbs 而不是套接字运行时，就是这个变量在选择它们使用哪些适配器——它是 [`NCCL_SOCKET_IFNAME`](#nccl_socket_ifname) 在 RDMA 下的对应变量。运行 `ibv_devinfo` 可以查看节点上有什么设备。

有三种形式：

```bash
export NCCL_IB_HCA=mlx5                # 名称以 mlx5 开头的每张卡的每个端口
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1  # 仅 mlx5_0 和 mlx5_1 的端口 1
export NCCL_IB_HCA=^=mlx5_1,mlx5_4     # 除 mlx5_1 和 mlx5_4 之外的全部
```

关于如何用排除形式绕开单块有问题的适配器，一个完整的示例见[如何诊断 NCCL 多 GPU 与多节点连接问题](#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues)。完整文档见[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)。


### `NCCL_P2P_DISABLE` {#nccl_p2p_disable}

禁用 P2P 通信——例如，即使有 NVLink 也不会使用它，性能因此会慢得多。通常你不会想要这样，但在紧要关头，调试期间它有时会有用。


### `NCCL_SOCKET_IFNAME` {#nccl_socket_ifname}

如果你有多个网络接口，并且想指定使用其中某一个，这个变量非常有用。

`NCCL_SOCKET_IFNAME` 选择 NCCL 可以用于引导（bootstrap）以及经由 TCP/IP 运行的集合通信的套接字——它不选择 InfiniBand Verbs 适配器（那些由 [`NCCL_IB_HCA`](#nccl_ib_hca) 负责）。

例如，要把 NCCL 限制在以太网接口上：

```bash
NCCL_SOCKET_IFNAME=eth
```

这个环境变量有时可用于调试连接问题，比如说某个接口被防火墙挡住了，而其他接口可能没有，于是可以改用其他接口试试。又或者你不确定某个问题是与网络接口有关还是与别的东西有关，这时测试其他接口有助于排除"问题来自网络"这一可能。
