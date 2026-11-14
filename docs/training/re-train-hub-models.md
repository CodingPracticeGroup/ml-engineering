# 使用微调示例从头重新训练 HF Hub 模型 {#re-train-hf-hub-models-from-scratch-using-finetuning-examples}

HF Transformers 有一流的微调示例 https://github.com/huggingface/transformers/tree/main/examples/pytorch ，几乎覆盖任何模态，而且这些示例开箱即用。

**但如果你想要从头重新训练，而不是做微调，该怎么办呢。**

下面介绍一个简单的小技巧来做到这一点。

我们将使用 `facebook/opt-1.3b`，并在这里以 bf16 训练方案作为示例：

```bash
cat << EOT > prep-bf16.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

mname = "facebook/opt-1.3b"

config = AutoConfig.from_pretrained(mname)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(mname)

path = "opt-1.3b-bf16"

model.save_pretrained(path)
tokenizer.save_pretrained(path)
EOT
```

现在运行：

```bash
python prep-bf16.py
```

这会创建一个名为 `opt-1.3b-bf16` 的文件夹，其中包含了从头训练模型所需的一切。换句话说，你得到了一个类似预训练模型的模型，只不过它只完成了初始化，还没有进行任何训练。

如果你计划改用其他精度，请调整上面的脚本，使用 `torch.float16` 或 `torch.float32`。

现在你可以照常对这个保存下来的模型进行微调了：

```bash
torchrun \
--nproc_per_node=1 --nnode=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=9901 \
examples/pytorch/language-modeling/run_clm.py --bf16 \
--seed 42 --model_name_or_path opt-1.3b-bf16 \
--dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--gradient_accumulation_steps 1 --do_train --do_eval --logging_steps 10 \
--save_steps 1000 --eval_steps 100 --weight_decay 0.1 --num_train_epochs 1 \
--adam_beta1 0.9 --adam_beta2 0.95 --learning_rate 0.0002 --lr_scheduler_type \
linear --warmup_steps 500 --report_to tensorboard --output_dir save_dir
```

其中的关键参数是：
```
--model_name_or_path opt-1.3b-bf16
```

其中 `opt-1.3b-bf16` 是你在上一步中刚刚生成的本地目录。

有时候你能找到与原模型训练时使用的相同数据集，有时候则不得不使用替代数据集。

其余的超参数通常可以在随模型一同发布的论文或文档中找到。

总之，借助这个做法，你可以利用微调示例来重新训练你在 [HF hub](https://huggingface.co/models) 上能找到的任何模型。
