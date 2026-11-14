# 只用单个节点模拟多节点环境 {#emulate-a-multi-node-setup-using-just-a-single-node}

目标是在一个只有 2 块 GPU 的单节点上模拟出 2 节点的环境（用于测试目的）。当然，这一点还可以进一步扩展为[更大的配置](#larger-set-ups)。

这里我们使用 `deepspeed` 启动器。其实并不需要真正用到任何 deepspeed 代码，只是用它更高级的功能更省事。你只需要安装 `pip install deepspeed`。

完整的搭建步骤如下：

1. 创建一个 `hostfile`：

```bash
$ cat hostfile
worker-0 slots=1
worker-1 slots=1
```

2. 在你的 ssh 客户端中添加相匹配的配置

```bash
$ cat ~/.ssh/config
[...]

Host worker-0
    HostName localhost
    Port 22
Host worker-1
    HostName localhost
    Port 22
```

如果端口不是 22，请改成实际的端口；如果主机名不是 `localhost`，也请照此修改。


3. 由于你本地的环境很可能设置了密码保护，请务必把自己的公钥加入 `~/.ssh/authorized_keys`

`deepspeed` 启动器明确使用免密连接，例如在 worker0 上它会运行：`ssh -o PasswordAuthentication=no worker-0 hostname`，所以你可以随时用下面的命令来调试 ssh 的设置：

```bash
$ ssh -vvv -o PasswordAuthentication=no worker-0 hostname
```

4. 创建一个测试脚本，用来检查两块 GPU 是否都被用上。

```bash
$ cat test1.py
import os
import time
import torch
import deepspeed
import torch.distributed as dist

# 关键技巧：使用第 2 块 gpu（否则两个进程都会使用 gpu0）
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dist.init_process_group("nccl")
local_rank = int(os.environ.get("LOCAL_RANK"))
print(f'{dist.get_rank()=}, {local_rank=}')

x = torch.ones(2**30, device=f"cuda:{local_rank}")
print(x)
time.sleep(100)
```

运行：

```bash
$ deepspeed -H hostfile test1.py
[2022-09-08 12:02:15,192] [INFO] [runner.py:415:main] Using IP address of 192.168.0.17 for node worker-0
[2022-09-08 12:02:15,192] [INFO] [multinode_runner.py:65:get_cmd] Running on the following workers: worker-0,worker-1
[2022-09-08 12:02:15,192] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test1.py
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:02:16,517] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:02:16,518] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
worker-0: tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
```

如果 ssh 设置成功了，你可以并行运行 `nvidia-smi`，并观察到两块 GPU 都因为 `torch.ones` 调用而分配了约 4GiB 显存。

请注意，脚本里硬塞了一个 `CUDA_VISIBLE_DEVICES`，用来告诉第 2 个进程使用 gpu1，但两种情况下它都会被看作 `local_rank==0`。

5. 最后，让我们测试 NCCL 集合通信（collective）也能正常工作

该脚本改编自 [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py)，只对 `os.environ["CUDA_VISIBLE_DEVICES"]` 做了调整

```bash
$ cat test2.py
import deepspeed
import fcntl
import os
import socket
import time
import torch
import torch.distributed as dist

# 关键技巧：让第 2 个进程使用第 2 块 GPU（否则两个进程都会使用 gpu0）
if os.environ["RANK"] == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def printflock(*msgs):
    """ 解决多进程打印内容交错的问题 """
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            print(*msgs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
hostname = socket.gethostname()

gpu = f"[{hostname}-{local_rank}]"

try:
    # 测试分布式
    dist.init_process_group("nccl")
    dist.all_reduce(torch.ones(1).to(device), op=dist.ReduceOp.SUM)
    dist.barrier()
    print(f'{dist.get_rank()=}, {local_rank=}')

    # 测试 cuda 可用并且能分配显存
    torch.cuda.is_available()
    torch.ones(1).cuda(local_rank)

    # 全局 rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    printflock(f"{gpu} is OK (global rank: {rank}/{world_size})")

    dist.barrier()
    if rank == 0:
        printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        printflock(f"device compute capabilities={torch.cuda.get_device_capability()}")
        printflock(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")

except Exception:
    printflock(f"{gpu} is broken")
    raise
```

运行：

```bash
$ deepspeed -H hostfile test2.py
[2022-09-08 12:07:09,336] [INFO] [runner.py:415:main] Using IP address of 192.168.0.17 for node worker-0
[2022-09-08 12:07:09,337] [INFO] [multinode_runner.py:65:get_cmd] Running on the following workers: worker-0,worker-1
[2022-09-08 12:07:09,337] [INFO] [runner.py:504:main] cmd = pdsh -S -f 1024 -w worker-0,worker-1 export PYTHONPATH=/mnt/nvme0/code/huggingface/multi-node-emulate-ds;  cd /mnt/nvme0/code/huggingface/multi-node-emulate-ds; /home/stas/anaconda3/envs/py38-pt112/bin/python -u -m deepspeed.launcher.launch --world_info=eyJ3b3JrZXItMCI6IFswXSwgIndvcmtlci0xIjogWzBdfQ== --node_rank=%n --master_addr=192.168.0.17 --master_port=29500 test2.py
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=0
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-0: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:136:main] WORLD INFO DICT: {'worker-0': [0], 'worker-1': [0]}
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:142:main] nnodes=2, num_local_procs=1, node_rank=1
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:155:main] global_rank_mapping=defaultdict(<class 'list'>, {'worker-0': [0], 'worker-1': [1]})
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:156:main] dist_world_size=2
worker-1: [2022-09-08 12:07:10,635] [INFO] [launch.py:158:main] Setting CUDA_VISIBLE_DEVICES=0
worker-0: dist.get_rank()=0, local_rank=0
worker-1: dist.get_rank()=1, local_rank=0
worker-0: [hope-0] is OK (global rank: 0/2)
worker-1: [hope-0] is OK (global rank: 1/2)
worker-0: pt=1.12.1+cu116, cuda=11.6, nccl=(2, 10, 3)
worker-0: device compute capabilities=(8, 0)
worker-0: pytorch compute capabilities=['sm_37', 'sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86']
worker-1: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] Process 576485 exits successfully.
worker-0: [2022-09-08 12:07:13,642] [INFO] [launch.py:318:main] Process 576484 exits successfully.
```

好，任务完成。

我们验证了 NCCL 集合通信可用，但它们用的是本地 NVLink/PCIe，而不是真实多节点里的 IB/ETH 连接，所以它是否足够用于测试，取决于需要测试什么。


## 更大的配置 {#larger-set-ups}

现在，假设你有 4 块 GPU，想模拟 2x2 的节点。那么只要把 `hostfile` 改成：

```bash
$ cat hostfile
worker-0 slots=2
worker-1 slots=2
```
并把 `CUDA_VISIBLE_DEVICES` 的技巧改成：

```bash
if os.environ["RANK"] in ["2", "3"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
```

其他一切都保持不变。


## 自动化这一过程 {#automating-the-process}

如果你希望用自动化的方式来应对任意形状的拓扑，可以用类似这样的代码：

```python
def set_cuda_visible_devices():
    """
    通过调整 CUDA_VISIBLE_DEVICES 环境变量，为每个被模拟的节点自动分配正确的一组 gpu
    """

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    emulated_node_size = int(os.environ["LOCAL_SIZE"])
    emulated_node_rank = int(global_rank // emulated_node_size)
    gpus = list(map(str, range(world_size)))
    emulated_node_gpus = ",".join(gpus[emulated_node_rank*emulated_node_size:(emulated_node_rank+1)*emulated_node_size])
    print(f"Setting CUDA_VISIBLE_DEVICES={emulated_node_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = emulated_node_gpus

set_cuda_visible_devices()
```


## 用单块 GPU 模拟多块 GPU {#emulating-multiple-gpus-with-a-single-gpu}

下面要谈的需求与本文讨论的问题属于不同的维度，但彼此相关，所以我觉得在这里分享一些见解会有帮助：

在 NVIDIA A100 上，你可以用 [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)（Multi-Instance GPU，多实例 GPU）在仅仅一块真实 GPU 上模拟出多达 7 个 GPU 实例，但多年来这些实例只能用于独立运行——例如你无法在这些 GPU 之间做 DDP 或任何 NCCL 通信。我曾希望可以用自己的 A100 模拟出 7 个实例，再加上一块真实 GPU，从而拥有 8 块 GPU 来做开发——结果不行，它做不到。当时我向 NVIDIA 的工程师询问此事，得到的答复是没有计划支持这种用例。

这一点已经改变了——[NCCL 2.31（2026）加入了对跨越 MIG 实例的通信器（communicator）的实验性支持](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-mig-instances)，因此单块物理 GPU 现在可以代替一个小型多 GPU 节点，你可以在它的各个实例上运行 DDP、FSDP、ZeRO 以及任何 NCCL 集合通信。下面给出的是一套在 H200 上可行的做法，以及对这些被模拟的 GPU 之间实际互访速度的测量结果。


### 获取足够新的 NCCL {#getting-a-new-enough-nccl}

你需要一个支持 CUDA 13.0 或更新版本的 NVIDIA 驱动，以及 NCCL 2.31 或更新版本。

截至 2026 年 8 月，PyTorch 最新的 2.13.0 自带的版本较低。在出现内置 NCCL>=2.31 的 PyTorch 构建版之前，请自己安装 NCCL，并让它优先于自带的那一份被加载：

```bash
pip install -U --target $HOME/nccl-new 'nvidia-nccl-cu13>=2.31'
LD_PRELOAD=$HOME/nccl-new/nvidia/nccl/lib/libnccl.so.2 python train.py
```

它装到自己的目录里，而不是装进当前环境，因为 PyTorch 会锁定确切的 NCCL 版本（2.13.0 对应 `nvidia-nccl-cu13==2.29.7`），直接覆盖安装会让 pip 的依赖解析器与 torch 的元数据发生冲突。随后 `LD_PRELOAD` 会让动态链接器在触及 torch wheel 内部自带的那份库之前，先从新库里解析 NCCL 的符号。

为了确认这一点，请用 `NCCL_DEBUG=INFO` 运行一次程序，并检查横幅信息是否写着 `NCCL version 2.31.2+cuda13.3` 或更新的版本。不要用 `torch.cuda.nccl.version()` 来做这项检查——它报告的是 PyTorch 编译时所依据的版本，而不是真正在干活的那个被预加载的版本。


### MIG 需要的 NCCL 设置 {#the-nccl-settings-mig-needs}

```bash
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_NVLS_ENABLE=0
```

NCCL 文档说 `NCCL_MULTI_RANK_GPU_ENABLE` 并非必需，因为每个 MIG 实例都是一个独立的 CUDA 设备——但在 NCCL 2.31.2 的 H200 上，不设置它通信器就起不来，所以还是设置上吧。NCCL 是靠父 GPU 的 NVML 索引和 PCI 总线 id 来识别某个 rank 所在设备的，而不是靠 MIG 实例，因此同一块 GPU 的四个实例上的四个 rank，在它看来就是四个 rank 共享一块设备；没有这个变量时，初始化会停在 `Multiple Ranks are using the same GPU/Partition`。

`NCCL_NVLS_ENABLE=0` 是一道保险，而不是硬性要求。在这种情况下，NCCL 的默认值本来就不会启用 NVLink SHARP，所以如果你的环境里没有设置这个变量，可以不去管它。但许多集群镜像和 pod spec 会导出 `NCCL_NVLS_ENABLE=1`，而一旦明确强制要求使用 NVLink SHARP，NCCL 就会拒绝启动并给出 `NCCL_NVLS_ENABLE has been set to "1" and communicator has multiple ranks using the same NVML device`。把它设为 0 不会带来任何开销，而且无论环境里已经导出了什么，都能让这套做法正常工作。你也可以设 `NCCL_NVLS_ENABLE=2`，它会在可行时自动使用 NVLS。

文档还让你设置 `NCCL_MNNVL_ENABLE=0`。在没有 NVLink fabric 的节点上，这个设置不改变任何东西——同一个基准测试在它未设置、设为 `0` 和设为 `1` 时分别得到 142.28、142.42 和 142.23 GBps——所以它只在 GB200 NVL72 这类连接了 fabric 的系统上才有影响。


### 切分一块 GPU {#carving-up-a-gpu}

先列出这块 GPU 提供的配置文件（profile），然后在它上面启用 MIG，再创建实例。在 H200 上我们最多可以创建 7 个实例。我们在 GPU 4 上创建 4 个各 35GB 的实例，这样就可以在单块 GPU 上做例如 SP=2、EP=2 的模型并行。

```bash
nvidia-smi mig -lgip             # 显示这块 GPU 上可用的配置文件
sudo nvidia-smi -i 4 -mig 1      # 在 GPU 4 上启用 MIG
sudo nvidia-smi mig -i 4 -cgi 1g.35gb,1g.35gb,1g.35gb,1g.35gb -C
```

若要撤销 MIG 并重新获得完整的 GPU：

```bash
sudo nvidia-smi mig -i 4 -dci && sudo nvidia-smi mig -i 4 -dgi
sudo nvidia-smi -i 4 -mig 0
```


### 查找 MIG 的 UUID {#finding-the-mig-uuids}

必须通过 `CUDA_VISIBLE_DEVICES` 给每个 rank 恰好一个实例，而 MIG 实例是按 UUID 寻址的，不是按索引。`nvidia-smi -L` 会把它们缩进列在各自父 GPU 的下面：

```bash
$ nvidia-smi -L
[...]
GPU 4: NVIDIA H200 (UUID: GPU-31f70a1b-dee6-617e-13c5-5c2a2fbe0013)
  MIG 1g.35gb  Device 0: (UUID: MIG-e26b1e11-8424-5ef0-97a8-d87e519f7824)
  MIG 1g.35gb  Device 1: (UUID: MIG-1a7d7808-5dcc-50fa-842c-2ceaa9f6e8ee)
  MIG 1g.35gb  Device 2: (UUID: MIG-46c30315-dfa5-56d4-82f6-345f84b4d52d)
  MIG 1g.35gb  Device 3: (UUID: MIG-6b4ff05d-689c-5bf3-8a51-3ff519c617ac)
```

把节点上每一块尺寸与 rank 相当的设备收集到一个以逗号分隔的变量里——处于 MIG 模式的 GPU 贡献它的各个实例，不处于 MIG 模式的 GPU 贡献它自己：

```bash
export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')
```


### 启动各个 rank {#launching-the-ranks}

`torchrun` 会给每个进程一个 `LOCAL_RANK`，而这正是给每个进程分配各自实例所需要的全部内容：

```bash
torchrun --nproc_per_node 4 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python train.py'
```

必须把 `LOCAL_RANK` 重置为 `0`，参见[注意事项](#caveats)。

`all_reduce`、`all_gather`、`reduce_scatter`、`broadcast` 以及一个 DDP 训练步骤，都能在这样 4 个实例之间正常工作。



### 这些被模拟的 GPU 之间互访有多快？ {#how-fast-do-the-emulated-gpus-talk-to-each-other}

只有当集合通信不慢得离谱时，这种模拟才有用处，所以这里给出的是 [torch-dist-bench.py](../network/benchmarks/torch-dist-bench.py) 在一块 H200 的 MIG 实例上的运行结果，并与相同数量整块 H200 的测量结果作对照——4 个实例对比 4 块真实 GPU，2 个实例对比 2 块真实 GPU。

下面用到的配置文件全部在一块 H200 141GB 上：

| MIG 配置文件 |   计算切片数 | 显存 | 每块 GPU 的实例数 |
| :----------- | -----------: | ---: | ----------------: |
| `1g.18gb`    |   7 份中的 1 份 | 18GB |                 7 |
| `1g.35gb`    |   7 份中的 1 份 | 35GB |                 4 |
| `2g.35gb`    |   7 份中的 2 份 | 35GB |                 3 |
| `3g.71gb`    |   7 份中的 3 份 | 71GB |                 2 |

下面每一张结果表都以它作为对照基准的整卡那一行开头，然后按 `busbw` 从大到小列出各 MIG 行。

4 个 rank，512MiB 负载：

| 这 4 个 rank 运行在什么上    |     busbw | 与整卡相比 |
| :--------------------------- | --------: | ---------: |
| 4 块整卡 H200（NVLink）      | 341.1GBps |       100% |
| 同一块 GPU 上的 4 个 `1g.35gb` | 273.8GBps |        80% |
| 同一块 GPU 上的 4 个 `1g.18gb` | 142.3GBps |        42% |

4 个 rank，2GiB 负载：

| 这 4 个 rank 运行在什么上    |     busbw | 与整卡相比 |
| :--------------------------- | --------: | ---------: |
| 4 块整卡 H200（NVLink）      | 354.7GBps |       100% |
| 同一块 GPU 上的 4 个 `1g.35gb` | 281.3GBps |        79% |
| 同一块 GPU 上的 4 个 `1g.18gb` | 144.7GBps |        41% |

2 个 rank，512MiB 负载：

| 这 2 个 rank 运行在什么上    |     busbw | 与整卡相比 |
| :--------------------------- | --------: | ---------: |
| 2 块整卡 H200（NVLink）      | 308.2GBps |       100% |
| 同一块 GPU 上的 2 个 `3g.71gb` | 413.8GBps |       134% |
| 同一块 GPU 上的 2 个 `2g.35gb` | 245.3GBps |        80% |
| 同一块 GPU 上的 2 个 `1g.35gb` | 230.9GBps |        75% |
| 同一块 GPU 上的 2 个 `1g.18gb` | 122.0GBps |        40% |

2 个 rank，2GiB 负载：

| 这 2 个 rank 运行在什么上    |     busbw | 与整卡相比 |
| :--------------------------- | --------: | ---------: |
| 2 块整卡 H200（NVLink）      | 331.7GBps |       100% |
| 同一块 GPU 上的 2 个 `3g.71gb` | 429.5GBps |       129% |
| 同一块 GPU 上的 2 个 `2g.35gb` | 250.4GBps |        75% |
| 同一块 GPU 上的 2 个 `1g.35gb` | 235.0GBps |        71% |
| 同一块 GPU 上的 2 个 `1g.18gb` | 123.6GBps |        37% |

在上面那张配置文件表中，计算切片数就是配置文件名里的那个数字（`2g.35gb` 拿到这块 GPU 的 7 个计算切片中的 2 个），而显存那一列说明该实例拥有 8 个显存切片中的几个——18GB 是 1 个切片，35GB 是 2 个，71GB 是 4 个。

测试环境：H200 141GB SXM、`torch==2.9.1+cu130`、如上所述预加载的 NCCL 2.31.2、CUDA 驱动 13.0。

这里最大的意外，是配置文件的选择影响有多大。MIG 实例之间的流量从不离开这块 GPU——NCCL 把每一个 channel 都记录为 `P2P/CUMEM`，整个运行过程中没有任何 socket 或 PCIe 回退——所以限制在于该实例所拥有的显存带宽，而这由它的显存切片决定，几乎与它的 SM 无关。两张 512MiB 的表把这一点单独分离了出来：在 4 个 rank 那张表里，从 `1g.18gb` 换到 `1g.35gb`，计算保持 1 个切片不变而显存翻倍，带宽几乎翻倍（142 -> 274GBps）；而在 2 个 rank 那张表里，从 `1g.35gb` 换到 `2g.35gb`，显存不变而计算翻倍，却只换来 6% 的提升（231 -> 245GBps）。

因此，4 个 `1g.35gb` 实例能重现 4 块真实 H200 之间约 79% 的互联带宽，而 4 个 `1g.18gb` 实例——rank 数量相同、切片更小——只有 41%。请挑选在每个实例上显存最多、同时又能给出你所需要的 rank 数量的那种配置文件。

而在只有 2 个 rank 时，`3g.71gb` 实例实际上比 2 块整卡 GPU *更快*（在 2 个 rank、2GiB 那张表里是 429 对 332GBps），因为在单块芯片自己的 HBM 内部做一次拷贝，胜过一次跨 NVLink 的跳转。

### 把实例分散到 2 块 GPU 上 {#spreading-the-instances-over-2-gpus}

在 NVIDIA 解决掉那个奇怪的[每块 GPU 只有 7 个实例的限制](#only-7-instances-per-gpu)之前，如果你想做 [3D 并行](model-parallelism/README.md#dppptp)，至少需要 8 块 GPU 或实例。所以眼下我们不得不使用 2 块 GPU 来凑够 8 个。

以上所有内容都把整个被模拟的世界放在一块物理 GPU 内部。把 2 块 GPU 各切成 4 个实例就得到 8 个 rank，其设置与基准测试如下：

```bash
for i in 0 1; do
    sudo nvidia-smi -i $i -mig 1
    sudo nvidia-smi mig -i $i -cgi 1g.35gb,1g.35gb,1g.35gb,1g.35gb -C
done

export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')

torchrun --nproc_per_node 8 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python torch-dist-bench.py --payload_size_in_gib 2'
```

8 个 rank，512MiB 负载：

| 这 8 个 rank 运行在什么上   |     busbw | 与整卡相比 |
| :-------------------------- | --------: | ---------: |
| 8 块整卡 H200（NVLink）     | 450.2GBps |       100% |
| 2 块 GPU 上各 4 个 `1g.35gb` |  26.8GBps |         6% |

8 个 rank，2GiB 负载：

| 这 8 个 rank 运行在什么上   |     busbw | 与整卡相比 |
| :-------------------------- | --------: | ---------: |
| 8 块整卡 H200（NVLink）     | 465.9GBps |       100% |
| 2 块 GPU 上各 4 个 `1g.35gb` |  25.6GBps |         5% |

这是一次崩塌，而且不是小消息造成的影响——2GiB 的负载并不比 512MiB 的更好。NCCL 自己的日志说明了原因：在这个环形的众多 channel 中，有 24 跳走的是 `via P2P/CUMEM`，另有 6 跳走的是 `via SHM/direct`。留在单块 GPU 内部的那些跳保留了快速路径，但跨到另一块 GPU 的那些跳退回到了主机共享内存，因为 CUDA 驱动不支持 MIG 实例之间的 NVLink P2P——这与这里无法使用 NVLink SHARP 是同一个限制。一个环形拓扑按它最慢的那条链路的速度运行，所以那几跳主机内存的速度决定了全部 8 个 rank 的节奏。


### 混合使用 MIG 实例与完整 GPU {#mixing-mig-instances-and-full-gpus}

不必把所有 GPU 都切分。由于 7 个实例已经是一块 GPU 能给你的上限，凑出 8 个 rank 的世界最省钱的办法，是把一块 H200（或另一块高端 GPU）切到最多的 7 个实例，而把第二块 GPU 保持完整。上面那条 `MIG_UUIDS` 命令对这种混合情形已经能生成正确的列表——被切分的 GPU 贡献它的 7 个实例，完整的那块贡献它自己：

```bash
sudo nvidia-smi -i 0 -mig 1
sudo nvidia-smi mig -i 0 -cgi 1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb,1g.18gb -C

export MIG_UUIDS=$(nvidia-smi -L | perl -ne 'if (/\(UUID: (GPU-[^)]+)\)/) { push @u, $1 } elsif (/\(UUID: (MIG-[^)]+)\)/) { my $m = $1; pop @u if $u[-1] =~ /^GPU-/; push @u, $m } END { print join ",", @u }')

torchrun --nproc_per_node 8 --no-python bash -c '
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((LOCAL_RANK+1)))
export LOCAL_RANK=0
exec python torch-dist-bench.py --payload_size_in_gib 2'
```

8 个 rank，512MiB 负载：

| 这 8 个 rank 运行在什么上              |     busbw | 与整卡相比 |
| :------------------------------------- | --------: | ---------: |
| 8 块整卡 H200（NVLink）                | 450.2GBps |       100% |
| 同一块 GPU 上的 7 个 `1g.18gb` 加 1 块整卡 H200 |  28.5GBps |         6% |

8 个 rank，2GiB 负载：

| 这 8 个 rank 运行在什么上              |     busbw | 与整卡相比 |
| :------------------------------------- | --------: | ---------: |
| 8 块整卡 H200（NVLink）                | 465.9GBps |       100% |
| 同一块 GPU 上的 7 个 `1g.18gb` 加 1 块整卡 H200 |  27.5GBps |         6% |

完整的那块 GPU 什么也救不回来：24 跳在被切分的 GPU 内部以 `via P2P/CUMEM` 运行，另外 6 跳以 `via SHM/direct` 走向完整的那块，环形拓扑再次按主机内存那一跳的速度运行。由此得到的有用推论是：第二个设备自身的带宽并不重要——第八个位置放一块完整 H200 得到 28.5GBps，并不比第八个 rank 是另一块被切分 H200 上的 MIG 实例的那次运行（26.8GBps）更好。所以，如果目标只是越过每块 GPU 7 个实例的上限，第二个设备可以是任何 CUDA 能与之通信的 GPU，而且它自己不需要支持 MIG，因为它是整块加入的——一块 H200 加上一块规格低一些的 GPU，就是一个 8 个 rank 的世界。

实用规则：把被模拟的多 GPU 作业保持在单块物理 GPU 内部。跨到第二块 GPU，相比在同一块 GPU 上放 4 个实例要付出 10 倍的代价（512MiB 时 26.8 对 273.8GBps），而且只剩下 8 块真实 GPU 所能达到的 5%。不过如果你用的是很小的模型和很小的负载，就开发目的而言大概影响不大。

请务必记住，这模拟的是多 GPU 作业的*拓扑*，而不是它的性能——每个 rank 还只分到一部分 SM 和一部分显存，所以它是一个开发用的、测试用的工具，不是衡量吞吐量的手段。


### 注意事项 {#caveats}

#### 每个 rank 中的 `LOCAL_RANK` 都是 `0` {#local_rank-is-0-in-every-rank}

一旦 `CUDA_VISIBLE_DEVICES` 指定了单个实例，该进程就只能看到一块设备，而 CUDA 会把它编号为 `0`，因此按 torchrun 给出的值去做 `torch.cuda.set_device(local_rank)` 的代码会失败并报 `invalid device ordinal`。没有办法绕过它：即使有多个 MIG 设备可见，CUDA 每个进程也只枚举一个计算实例。当各 rank 横跨 2 块被切分的 GPU 时，这一点依然成立——某个 rank 落在哪一块物理 GPU 上是由 UUID 承载的，而不是由序号承载，所以每个 rank 都说自己用的是 `cuda:0`，而每个 rank 其实都在不同的硬件上。

这就使得每个 rank 上报的都是 `LOCAL_RANK=0`，这一点值得说清楚。

`torch.distributed` 里的任何东西都不受影响。`dist.get_rank()`、`dist.get_world_size()` 和进程组是根据 `RANK` 和 `WORLD_SIZE` 建立的，而启动包装脚本并没有改动它们。常用的设备选择写法也照常有效：

```python
local_rank = int(os.environ["LOCAL_RANK"])   # 在每个进程里都是 0
torch.cuda.set_device(local_rank)            # 绑定到本 rank 自己的那个实例
model = DDP(model.to(local_rank), device_ids=[local_rank])
```

每个进程都请求 0 号设备，而每个进程拿到的却是不同的切片，因为各 rank 的 `CUDA_VISIBLE_DEVICES` 是不同的。

会被破坏的是那些把 `LOCAL_RANK` 当作身份标识、而不是当作设备序号来使用的代码，其中最常见的是节点主进程守卫：

```python
if local_rank == 0:
    download_dataset()   # 本意是每个节点只运行一次
```

在 MIG 下，这个条件在每个进程里都成立，于是所有 rank 都会去下载数据集、写同一个日志文件，或者争抢着去填充同一个缓存目录。

在单节点模拟中，解决办法是改用 `RANK`，它在整个作业内仍然是唯一的：

```python
if dist.get_rank() == 0:
    download_dataset()
```

如果这段代码必须不经修改也能在真实的多节点运行中正常工作，那就换用另一个名字保留原来的槽位号。启动包装脚本会在覆盖它之前先把它存下来：

```bash
torchrun --nproc_per_node 8 --no-python bash -c '
export MIG_SLOT=$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=$(echo $MIG_UUIDS | cut -d, -f$((MIG_SLOT+1)))
export LOCAL_RANK=0
exec python train.py'
```

而守卫代码读取其中存在的那个变量：

```python
node_slot = int(os.environ.get("MIG_SLOT", os.environ["LOCAL_RANK"]))
if node_slot == 0:
    download_dataset()
```

#### 每块 GPU 只有 7 个实例 {#only-7-instances-per-gpu}

MIG 是沿着芯片硅片上本来就存在的边界来切的。一个计算切片来自一个 GPC（Graphics Processing Cluster，图形处理簇），它是把一组 SM 连同自己的工作分发机制打包在一起的顶层单元；一个显存切片则来自与 HBM 堆栈相连的显存子系统。据 NVIDIA 方面的消息，Ampere、Hopper、Blackwell 以及近期的 Rubin，其完整的 GPU 实现中各含正好 8 个 GPC（不过实际出货的 SKU 可能启用得更少）。

MIG 最多能有 7 个实例而不是 8 个，因为第 8 个计算单元被保留用来管理这些实例——一旦启用 MIG 模式它就被留出来，无论你随后是创建 7 个实例还是 1 个。显存子系统没有这样的占用者，所以 8 个显存切片全都可用，MIG 文档也直接给出了由此产生的比例：每个显存切片是显存的 1/8，每个 SM 切片是 SM 的 1/7。

这种不对称正是上面那张配置文件表所展示的。7 个 `1g` 实例用掉了全部 7 个计算切片，却只用掉 8 个显存切片中的 7 个，因此在一块被完全切分的 GPU 上，会有 1 个切片的 HBM 被搁置——H200 141GB 中的 18GB。这也是为什么 `4g` 配置文件只允许 1 个实例：再来一个就需要 8 个计算切片，而只有 7 个可用。

#### `torch.cuda.device_count()` 与 `LOCAL_WORLD_SIZE` 不一致 {#torchcudadevice_count-disagrees-with-local_world_size}

一个进程只能看到自己那个实例，所以在 MIG 下每个 rank 的 `torch.cuda.device_count()` 都返回 1。在整块 GPU 上，这个调用恰好等于该节点上的 rank 数量，因此很多代码都依赖它——用来确定分片数量、用来构造要遍历的设备列表，或者用来检查请求的 world 大小能否放进这个节点。请改从环境变量里取这个数字：

```python
ranks_on_this_node = torch.cuda.device_count()               # MIG 下每个 rank 都是 1
ranks_on_this_node = int(os.environ["LOCAL_WORLD_SIZE"])     # torchrun 收到的 --nproc_per_node 的值
```

`LOCAL_WORLD_SIZE` 由 torchrun 在每个 worker 里设置，与 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE` 一起，并且在 MIG 下它仍然报告完整的本地 rank 数量。
