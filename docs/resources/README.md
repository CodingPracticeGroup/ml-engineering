# 资源 {#resources}

## 类似的在线指南 {#similar-online-guides}

- Boris Dayma 撰写了 [A Recipe for Training Large Models](https://wandb.ai/craiyon/report/reports/Recipe-Training-Large-Models--VmlldzozNjc4MzQz)（训练大模型的配方）

- HuggingFace 团队发表了 [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)（超大规模训练手册：在 GPU 集群上训练大语言模型）。

- Jacob Austin 维护着 [How to Scale Your Model: A Systems View of LLMs on TPUs](https://jax-ml.github.io/scaling-book/)（如何扩展你的模型：从系统视角看 TPU 上的大语言模型）

- Harvard Edge 团队出版了 [Machine Learning Systems](https://github.com/harvard-edge/cs249r_book)（机器学习系统；CS249r 教材；网站：[mlsysbook.ai](http://mlsysbook.ai/)）。


## 有用的资料汇编 {#useful-compilations}

- [@StellaAthena](https://github.com/StellaAthena) 制作了 [Common LLM Settings](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0) 电子表格（常见 LLM 训练配置汇总），当你即将开始一次新的 LLM 训练时，它会是一份极其有用的资料——因为它告诉你目前已知的 LLM 训练都有哪些、是怎么训练出来的。

- 几年前，我开始收集关于[这些模型是用哪种数据类型（dtype）训练的](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671)的信息——里面只收录了少数几个模型，但如果你在做关于数据类型的研究，它仍然有用。我当时想用这些信息来尝试编写一个[模型预训练 dtype 自动检测](https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb)工具，这里还有一篇相关的 [float16 与 bfloat16 数值特性对比](https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb)。


## 公开可获取的 LLM/VLM 训练日志 {#publicly-available-training-llmvlm-logbooks}

LLM/VLM（视觉语言模型）的训练日志、训练纪事和技术报告，是学习如何应对训练不稳定、基础设施与扩展挑战、以及如何挑选好的超参数等方面最好的资料来源之一。这里较早的条目是在一次训练运行期间记录的日志/纪事；而越来越多的情况下，同样这些来之不易的经验教训会作为技术报告中的专门章节出现——章节标题通常类似于“基础设施、扩展与效率”（Infrastructure, Scaling, and Efficiency）或“工程挑战与解决方案”（Engineering Challenges and Solutions），而且往往会谈及那些很少会出现在模型卡（model card）中的可靠性与运维挑战。

如果你知道有某个未列入此清单的公开 LLM/VLM 训练日志，或者有包含这类章节的技术报告，请通过 [Issue](https://github.com/stas00/ml-engineering/issues) 告诉我，或者通过 PR 添加进来。谢谢！

下面的清单除了按年份分组之外，没有任何特定顺序。

### 2021 {#2021}

- BigScience pre-BLOOM 108B 训练实验（2021）：
[训练纪事](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md) |
[完整规格与讨论](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)
（备份：
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md) |
[2](https://github.com/stas00/bigscience-backup/tree/master/train/tr8-104B-wide))


### 2022 {#2022}

- BigScience BLOOM-176B（2022）：
[训练纪事·前篇](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[训练纪事](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) |
[完整规格与讨论](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml)
（备份：
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md) |
[3](https://github.com/stas00/bigscience-backup/tree/master/train/tr11-176B-ml))

- Meta OPT-175B（2022）：
    [训练日志](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) | [视频](https://www.youtube.com/watch?v=p9IxoSkvZ-M)（备份：[1](https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles)）

- THUDM GLM-130B（2022）：[英文训练日志](https://github.com/zai-org/GLM-130B/blob/main/logs/main-log-en.md) | [中文版](https://github.com/zai-org/GLM-130B/blob/main/logs/main-log.md)（备份： [1](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md) | [2](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md)）


### 2023 {#2023}

- HuggingFace IDEFICS-80B 多模态模型（Flamingo 复现）（2023）：[学习日志](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) | [训练纪事](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md)（备份：[1](https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md) | [2](https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md)）

- BloombergGPT 50B LLM——见 [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)（面向金融领域的大语言模型）一文中的附录 C


### 2024 {#2024}

- [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627)（MegaScale：将大语言模型训练扩展到超过 10,000 块 GPU）——这篇论文探讨了各种训练问题及其解决办法——虽然涉及的模型是专有的，但同样具有指导意义/实用价值。

- Imbue 的 [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/blog/70b-infrastructure)（从裸金属到 70B 模型：基础设施搭建与脚本）是一篇非常详细的技术文章，讲述了他们在训练一个专有的 70B 参数模型过程中不得不克服的许多与训练相关的问题。

- Meta 的 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)（Llama 3 模型家族）用 [§3.3 基础设施、扩展与效率](https://arxiv.org/pdf/2407.21783#page=12)一节专门介绍训练技术栈，其中包括 §3.3.4 可靠性与运维挑战——特别是环境因素（昼夜温差，以及数万块 GPU 波动变化的功耗对数据中心电网的影响）如何干扰训练吞吐量，这一维度我在其他报告中还没有见到过记载。


### 2025 {#2025}

- Project Apertus（ETH Zürich、EPFL、CSCS）——[Apertus v1 Technical Report: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233)（Apertus v1 技术报告：让开放且合规的大语言模型惠及全球语言环境）——一套完全开放的 8B/70B 模型系列，使用最多 4096 块 GPU，在 CSCS 的 Alps 超算上针对 1800 多种语言的 15T 词元进行了训练。其中的“基础设施、扩展与效率”和“工程挑战与解决方案”章节（约 5 页）读起来就像这次训练运行的日志。


### 2026 {#2026}

- [Supercharging Olmo-core for Efficient and Scalable MoE Training](https://allenai.org/papers/olmocore3)（为高效可扩展的 MoE 训练全面强化 Olmo-core）——面向 2026 年大规模 MoE（混合专家）模型训练的实用建议！
- [Marin 535B-A23B live](https://mtracker.oa.dev/hero-run-535b)（Marin 535B-A23B 训练实况）——每条简短的记录都链接到一个详细的 GitHub Issue，你可以在那里阅读详情。已完成的训练复盘报告：[8B](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/)、[32B](https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/)
- Open Athena 的 [Architecture Decisions for Marin 535B-A23B](https://openathena.ai/blog/marin-535b-architecture/)（Marin 535B-A23B 的架构决策）——阐述了 535B-A23B MoE 模型每个组件背后的推理过程和降低风险的实验，该模型已完成在 18T 词元上的预训练。内容涉及：在 4k 到 65k 上下文扩展期间专家并行（expert parallelism）下的词元丢弃（token dropping）、Hyperball 优化器下嵌入的梯度尖峰，以及为什么为了稳定性选择固定的 QK-norm 乘数和较浅的 48 层模型，并附有各个实验对应的 GitHub Issue 链接。
- Project Apertus——[Apertus v1.5 Technical Report: From Open Text Models to Open Multimodal Reasoners](https://arxiv.org/abs/2610.15542)（Apertus v1.5 技术报告：从开放文本模型到开放多模态推理模型）——Apertus 8B/70B 模型的多模态继续预训练；与 v1 一样，报告包含“基础设施、扩展与效率”和“工程挑战与解决方案”章节（约 5 页）。

## 硬件搭建日志 {#hardware-setup-logbooks}

- Imbue 发布了一份详细的日志，记录了他们如何搭建起一个 512 节点的 IB 胖树（fat-tree）集群并让它正常运转：[From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/blog/70b-infrastructure)（从裸金属到 70B 模型：基础设施搭建与脚本），他们还将过程中打造的[集群工具](https://github.com/imbue-ai/cluster-health)开源了。

- SemiAnalysis 发表了一篇非常详尽的文章，讲述[搭建一个 Neocloud 集群需要什么](https://newsletter.semianalysis.com/p/ai-neocloud-playbook-and-anatomy)。
