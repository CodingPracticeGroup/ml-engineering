# ML Engineering 简体中文翻译规范（所有翻译代理必读并严格遵守）

## 任务与读者
把《Machine Learning Engineering》(Stas Bekman 著) 的英文 Markdown 忠实翻译成简体中文。
目标读者：计算机相关专业大学一年级学生。他们有基础编程能力，但没有分布式训练/ML 系统经验。
因此：语言要自然、通顺、易懂，长英文句拆成符合中文习惯的短句；专业术语**首次**出现时如中文不常见，用括号给出简短解释或保留英文原词。

## 总原则
1. **完整翻译，不摘要、不省略、不添加**。原文每一段、每一句、表格每一格、脚注、图片 alt 文本都要处理。不得增删观点，不得改写含义。
2. 语气与原文一致：作者用第一人称“我”的，译文保留“我”；不确定表述（可能、通常、据我经验）不得改成绝对表述。
3. 用词准确优先于行文漂亮。术语遵循下方术语表；术语表未覆盖且业界通用英文的，保留英文。

## 必须原样保留（不翻译）
- 所有代码块、命令、命令行参数、环境变量、日志输出、报错信息、文件路径、URL、邮箱。代码块内注释**要翻译**（保持注释语法）；但属于程序输出的示例输出不翻译。
- 行内代码 `like_this`、产品/库/工具/硬件名（PyTorch, CUDA, NCCL, InfiniBand, NVLink, Slurm, Kubernetes, Weights & Biases, H100, A100, GB200 等）、机构名、人名、GitHub 用户名。
- HTML 标签、注释 `<!-- -->`、badge、脚本、图片引用路径 `![](...)` 中的路径部分（alt 文本要翻译）。
- Markdown 结构：标题层级、表格结构、列表层级、引用块、脚注定义、链接 URL。

## 链接与锚点（最重要的一致性规则）
1. 相对链接路径（如 `../storage/README.md`、`./users.md`）和 URL **一律原样保留**。
2. 每个标题翻译后，在标题行末尾加 `{#原英文标题生成的锚点}`，锚点按 GitHub 规则由**原英文标题**生成：全部小写；删除除字母、数字、连字符、下划线以外的字符（如 `:` `(` `)` `,` `?` `.` `'` 直接删掉）；空格变连字符，连续空格变一个连字符。例：
   `## Troubleshooting slow jobs` → `## 排查运行缓慢的作业 {#troubleshooting-slow-jobs}`
   `### Tip: use "strace"` → `### 提示：使用 strace {#tip-use-strace}`
   若原标题已自带 `{#...}`，沿用该原有 id，不要重新生成。
3. 正文中 `#锚点` 形式的站内链接和 `../x/README.md#锚点` 的 URL/锚点部分保持原样，因为规则 2 保证锚点不变。文内目录（TOC）列表：条目文字翻译，链接目标原样保留。

## 排版
- 使用简体中文标点（，。：；“”（）），中英文之间、中文与数字之间加一个半角空格；全角标点前后不加空格。
- 保留原文数字、单位、表格对齐；单位名不翻译（GB、GiB、TFLOPS、Gbps、ns 等）。
- 章节标题（H1）翻译后保留原层级。

## 术语表（统一译法；标“保留”的保留英文）
大语言模型 LLM；预训练 pretraining；微调 fine-tuning；训练/推理（inference）；检查点 checkpoint；数据并行/张量并行/流水线并行/上下文并行/专家并行（data/tensor/pipeline/context/expert parallelism）；3D/4D 并行 保留；ZeRO 保留；分片 shard；激活重计算 activation recomputation（又称 activation checkpointing 激活检查点，注意与模型 checkpoint 区分）；批次 batch、批次大小 batch size；梯度 gradient、梯度累积 gradient accumulation；损失 loss 保留中文“损失”；发散 divergence；不稳定 instability；吞吐量 throughput；延迟 latency；带宽 bandwidth；开销 overhead；节点 node；集群 cluster；作业 job；队列 queue；调度器 scheduler；资源配额 quota；账户 account；分区 partition（Slurm 语境译为“分区”并保留原名如 gpu 分区）；加速器/加速卡 accelerator（上下文清晰时可直接说 GPU）；显存 GPU memory；内存 memory；存储 storage；文件系统 filesystem；对象存储 object storage；挂载 mount；互联 interconnect；拓扑 topology；轨道 rail（rail-optimized 译为“轨道优化”）；超额订阅 oversubscription；NUMA、RDMA、GPUDirect 保留；集合通信 collective；带宽测试/基准测试 benchmark（作动词“跑基准测试”）；性能剖析 profiling；监控 monitoring；指标 metrics；挂起 hang（“进程卡死/挂起”）；死锁 deadlock；超时 timeout；堆栈跟踪 stack trace；调试 debug；故障容错 fault tolerance；抢占 preemption；抢占式实例 spot instance；按需实例 on-demand；云服务商 cloud provider；订阅 subscription；预留实例 reserved instance；预算 budget；SLA 保留；词元 token；分词器 tokenizer；数据集 dataset；数据加载器 dataloader；嵌入 embedding；混合专家 MoE；核 kernel（GPU kernel 译为“内核”）；驱动 driver；固件 firmware；精度 precision；FP16/BF16/FP32/FP8 保留；混合精度 mixed precision；下溢/上溢 underflow/overflow；饱和 saturation；softmax/logits 保留；温度 temperature（softmax 语境译为“温度”）；KV 缓存 KV cache；填充 padding；注意力 attention。
其他术语：首次出现给出“中文（English）”格式，之后用该中文。

## 交付方式
读取给你的源文件（或源文件的某个分块），把译文**完整**写入指定的目标路径（UTF-8）。目标文件较长时分多次写入（先写前半，再追加后半），中间不要停顿总结。除目标文件外不要修改任何其他文件。完成后仅回复：DONE + 目标路径 + 一句自查说明（确认段落/标题数量与源一致）。
