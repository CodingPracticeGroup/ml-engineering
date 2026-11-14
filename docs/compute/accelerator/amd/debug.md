# 排查 AMD GPU 问题 {#troubleshooting-amd-gpus}

XXX：本文还处于非常早期的阶段——正在收集各种工具与笔记

由于我们大多数人都非常熟悉 NVIDIA 的工具，我会尽可能给出与这些熟悉工具之间的对应关系。

## 工具 {#tools}

### HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES {#hip_visible_devices-cuda_visible_devices-rocr_visible_devices}

ROCm 不止有一个“可见设备”（visible devices）开关，因为隔离可以发生在软件栈的不同层级上。对于在 ROCm 上运行的 PyTorch，惯用的 CUDA 风格做法是在 HIP 这一层设置：

```bash
HIP_VISIBLE_DEVICES=0,1 python my-program.py
```

`CUDA_VISIBLE_DEVICES` 同样会被识别（在 AMD 上它是 HIP 的 CUDA 兼容别名）。

但这只会过滤 **HIP** 运行时暴露出来的设备。ROCr 仍然可以初始化每一块 GPU，而位于 ROCr 之上的非 HIP 客户端（OpenCL、AOMP、UCX 等）不受影响。

要想把设备从整个用户态 ROCm 软件栈中隐藏掉，就要在 **ROCr** 这一层过滤：

```bash
ROCR_VISIBLE_DEVICES=0,1 python my-program.py
```

`ROCR_VISIBLE_DEVICES` 还接受 UUID 字符串（HIP 的列表则只接受索引）。如果两者都设置了，HIP 只能看到 ROCr 保留为可见的那些设备，因此 HIP 的索引是相对于那组过滤后的设备而言的。

在 Linux 上，当你需要整个软件栈范围的隔离时，AMD 的隔离文档倾向于使用 `ROCR_VISIBLE_DEVICES`；对于常规的单一框架 PyTorch 工作，`HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` 就够了。参见 [GPU 隔离技术](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html)。

### rocm-smi {#rocm-smi}

`rocm-smi`（相当于 `nvidia-smi`）会以精简的形式显示所有 ROCm 加速器的状态。

例如下面是一台 8xMI300X 节点上的输出：
```bash
$ rocm-smi
========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  [Model : Revision]    Temp        Power     Partitions      SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
        Name (20 chars)       (Junction)  (Socket)  (Mem, Compute)
====================================================================================================================
0       [0x74a1 : 0x00]       45.0°C      173.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
1       [0x74a1 : 0x00]       41.0°C      179.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
2       [0x74a1 : 0x00]       47.0°C      180.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
3       [0x74a1 : 0x00]       45.0°C      178.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W   17%   0%
        AMD Instinct MI300X
4       [0x74a1 : 0x00]       45.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
5       [0x74a1 : 0x00]       43.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
6       [0x74a1 : 0x00]       45.0°C      175.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
7       [0x74a1 : 0x00]       43.0°C      176.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================
```

奇怪的是，它不显示真正的显存用量——只显示百分比，这不太实用。

一个便于实时观察更新的实用别名：
```bash
alias wr='watch -n 1 rocm-smi'
```

### rocminfo {#rocminfo}

`rocminfo`（相当于 `nvidia-smi -q`）会显示每块加速器的详细信息。

这个工具会同时显示 CPU 和 GPU 的信息

下面是 cpu0 和 gpu0 的一段输出（注意它把 CPU 从节点 0..1 开始编号，然后把 GPU 编为节点 2..9）：
```bash
$ rocminfo
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE
System Endianness:       LITTLE
Mwaitx:                  DISABLED
DMAbuf Support:          YES

==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD EPYC 9534 64-Core Processor
  Uuid:                    CPU-XX
  Marketing Name:          AMD EPYC 9534 64-Core Processor
  Vendor Name:             CPU
  Feature:                 None specified
  Profile:                 FULL_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        0(0x0)
  Queue Min Size:          0(0x0)
  Queue Max Size:          0(0x0)
  Queue Type:              MULTI
  Node:                    0
  Device Type:             CPU
  Cache Info:
    L1:                      32768(0x8000) KB
  Chip ID:                 0(0x0)
  ASIC Revision:           0(0x0)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2450
  BDFID:                   0
  Internal Node ID:        0
  Compute Unit:            128
  SIMDs per CU:            0
  Shader Engines:          0
  Shader Arrs. per Eng.:   0
  WatchPts on Addr. Ranges:1
  Features:                None
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    792303268(0x2f3996a4) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       TRUE
    Pool 2
      Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
      Size:                    792303268(0x2f3996a4) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       TRUE
    Pool 3
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
      Size:                    792303268(0x2f3996a4) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       TRUE
  ISA Info:
[...]

  Name:                    gfx942
  Uuid:                    GPU-ababaeeffecddc50
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Min Size:          64(0x40)
  Queue Max Size:          131072(0x20000)
  Queue Type:              MULTI
  Node:                    2
  Device Type:             GPU
  Cache Info:
    L1:                      16(0x10) KB
    L2:                      8192(0x2000) KB
  Chip ID:                 29857(0x74a1)
  ASIC Revision:           1(0x1)
  Cacheline Size:          64(0x40)
  Max Clock Freq. (MHz):   2100
  BDFID:                   50688
  Internal Node ID:        7
  Compute Unit:            304
  SIMDs per CU:            4
  Shader Engines:          32
  Shader Arrs. per Eng.:   1
  WatchPts on Addr. Ranges:4
  Coherent Host Access:    FALSE
  Features:                KERNEL_DISPATCH
  Fast F16 Operation:      TRUE
  Wavefront Size:          64(0x40)
  Workgroup Max Size:      1024(0x400)
  Workgroup Max Size per Dimension:
    x                        1024(0x400)
    y                        1024(0x400)
    z                        1024(0x400)
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    2048(0x800)
  Grid Max Size:           4294967295(0xffffffff)
  Grid Max Size per Dimension:
    x                        4294967295(0xffffffff)
    y                        4294967295(0xffffffff)
    z                        4294967295(0xffffffff)
  Max fbarriers/Workgrp:   32
  Packet Processor uCode:: 132
  SDMA engine uCode::      19
  IOMMU Support::          None
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 2
      Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 3
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
      Size:                    201310208(0xbffc000) KB
      Allocatable:             TRUE
      Alloc Granule:           4KB
      Alloc Alignment:         4KB
      Accessible by all:       FALSE
    Pool 4
      Segment:                 GROUP
      Size:                    64(0x40) KB
      Allocatable:             FALSE
      Alloc Granule:           0KB
      Alloc Alignment:         0KB
      Accessible by all:       FALSE
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
      Machine Models:          HSA_MACHINE_MODEL_LARGE
      Profiles:                HSA_PROFILE_BASE
      Default Rounding Mode:   NEAR
      Default Rounding Mode:   NEAR
      Fast f16:                TRUE
      Workgroup Max Size:      1024(0x400)
      Workgroup Max Size per Dimension:
        x                        1024(0x400)
        y                        1024(0x400)
        z                        1024(0x400)
      Grid Max Size:           4294967295(0xffffffff)
      Grid Max Size per Dimension:
        x                        4294967295(0xffffffff)
        y                        4294967295(0xffffffff)
        z                        4294967295(0xffffffff)
      FBarrier Max Size:       32
```

## 多 GPU 运行挂起或缓慢与 IOMMU {#hangs-or-slow-multi-gpu-runs-and-iommu}

如果一次多 GPU 的 ROCm 运行出现挂起（hang）或慢得几乎不动，宿主机（host）的 IOMMU 配置往往是罪魁祸首。正确的设置方式因平台而异——请对照你的硬件来设置，而不是盲目地禁用它：

- **较新的 Instinct（MI300X、MI325X、MI350X、MI355X）**：AMD 的系统验收（system-acceptance）指南要求在 BIOS 中**启用** IOMMU，并在内核命令行上以直通（pass-through）模式运行——即 `iommu=pt` 与 `amd_iommu=on` 一起使用（在 Intel 主机上则是 `intel_iommu=on`）。MI350X/MI355X（ROCm 7.0.1 及以上）明确列出了 `IOMMU enabled` + `iommu=pt`。
- **MI100 / MI200（MI210/MI250/MI250X）**：AMD 认证过的 BIOS 设置表通常把 NBIO 的 **IOMMU 设为 Disable**，只有对拥有 256 个及以上逻辑 CPU 的主机才特别注明需要 `iommu=pt`（这样 X2APIC 才能寻址到每个核心）。
- **旧系统的最后手段**：在一些更老的系统上，如果直通模式和 BIOS 默认设置都不起作用，历史上惯用的权宜之计是完全禁用——通过 `amd_iommu=off` 彻底关闭（或者用更温和的 `iommu=soft`）——参见 AMD 的[面向多 GPU 环境的 IOMMU 注意事项](https://community.amd.com/t5/knowledge-base/iommu-advisory-for-multi-gpu-environments/ta-p/477468)和这份[问题报告](https://github.com/stas00/ml-engineering/issues/1#issuecomment-1076830400)（那个具体案例实际上发生在 NVIDIA A6000 上）。这是一项侵入性较强的、系统范围的启动改动，而 AMD 当前的指南更倾向于直通模式，而不是完全禁用。

这些是在启动时生效的内核参数，需要写在 `/etc/default/grub` 中（例如 `GRUB_CMDLINE_LINUX="... iommu=pt"`；grub 配置文件的路径因操作系统而异），改完后还需要重新生成 grub 配置并重启——因此需要管理员权限。

上面 `rocminfo` 输出中每块 GPU 各自的 `IOMMU Support` 那一行，反映的正是运行时实际看到的情况。
