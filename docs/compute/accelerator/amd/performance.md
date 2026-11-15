# AMD GPU 性能 {#amd-gpus-performance}

由于我还没有机会在 AMD GPU 上做过任何正式的工作，这里暂时只分享一些链接。

- [AMD Instinct MI300X 系统优化](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi300x.html)
- [AMD Instinct MI300X 工作负载优化](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html)

## 剖析工具 {#profilers}

[omniperf](https://github.com/ROCm/rocprofiler-compute) - 面向 AMD 硬件的高级性能剖析与分析工具 - 例如，它可以绘制你的 AMD 加速器的 roofline 性能曲线，还能做许多其他事情。
