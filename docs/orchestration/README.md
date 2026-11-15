# 编排 {#orchestration}

目前有许多容器/加速器的编排（orchestration）方案——其中很多都是开源的。

到目前为止，我一直使用的是 SLURM 和 Kubernetes。

## SLURM {#slurm}

- [SLURM](slurm/README.md) - Simple Linux Utility for Resource Management（简单 Linux 资源管理工具），你几乎肯定能在大多数高性能计算（HPC）环境中见到它，而且大多数云服务商通常都支持它。它已经存在超过 20 年了

### SLURM on Kubernetes {#slurm-on-kubernetes}

以下这些框架可以让你在 Kubernetes 之上运行 SLURM：

- [Slinky](https://github.com/slinkyproject) - 由 SLURM 的创作者开发的框架。
- [Soperator](https://github.com/nebius/soperator) - 来自 Nebius。
- CoreWeave 从未将他们的 [SUNK](https://www.coreweave.com/products/sunk) 开源，因此只有其客户才能使用。

## Kubernetes {#kubernetes}

另一个最受欢迎的编排器是 Kubernetes：

- [Kubernetes](kubernetes/README.md) - 也被称为 K8s，是一个用于自动化部署、扩展和管理容器化应用程序的开源系统。这里有一篇不错的 [SLURM 与 K8s 的对比](https://web.archive.org/web/20250324222116/https://www.fluidstack.io/post/is-kubernetes-or-slurm-the-best-orchestrator-for-512-gpu-jobs)。

## 其他编排框架 {#other-orchestration-frameworks}

以下是各种不如 SLURM 和 K8s 流行、但依然非常强大的编排方案：

- [dstack](https://github.com/dstackai/dstack) 是一个轻量级的开源方案，可作为 Kubernetes 和 SLURM 的替代品，支持多云与本地部署，简化 AI 容器的编排。它原生支持 NVIDIA、AMD 和 TPU。
- [SkyPilot](https://github.com/skypilot-org/skypilot) 是一个可以在任意基础设施上运行 AI 和批处理工作负载的框架，提供统一的执行方式、更高的成本节约和更高的 GPU 可用性。
- [OpenHPC](https://github.com/openhpc/ohpc) 提供了部署和管理 HPC Linux 集群所需的多种常见预构建组件，包括资源置备（provisioning）工具、资源管理、I/O 客户端、运行时、开发工具、容器，以及各种科学计算库。
- [run.ai](https://www.nvidia.com/en-us/software/run-ai/) - 已被 NVIDIA 收购，并计划很快开源。
- [Docker Swarm](https://docs.docker.com/engine/swarm/) 是一个容器编排工具。
- [IBM Platform Load Sharing Facility (LSF)](https://www.ibm.com/products/hpc-workload-management) Suites 是一个面向分布式高性能计算（HPC）的工作负载管理平台和作业调度器。
