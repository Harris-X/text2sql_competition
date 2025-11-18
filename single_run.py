import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- 1. 配置模型和数据集路径 ---
model_id = "/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct"
# 请将这里替换为您的 qwen3_spider_mysql 数据集路径
# 假设它是一个 JSON Lines 文件 (jsonl)
dataset_path = "/path/to/your/qwen3_spider_mysql.jsonl" 
# 假设您的数据集中用于训练的文本字段叫做 "text"
# (SFTTrainer 期望一个包含完整 "指令->回答" 格式化文本的字段)
# 您需要自己预处理数据，使其符合 qwen3_nothink 模板的格式
dataset_text_field = "text" 

# --- 2. QLoRA (4-bit 量化) 配置 ---
# 30B 模型必须使用量化，否则显存会爆炸
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 3. 加载模型和分词器 ---
print(f"正在加载模型: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # 自动将模型分布到所有可用 GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f"正在加载分词器: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Qwen3-Coder 没有默认 pad token，使用 eos token 作为 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# --- 4. 配置 LoRA (PEFT) ---
# 关键：为 MoE 模型指定正确的 target_modules，绝对不能用 'all'
lora_config = LoraConfig(
    r=8,                # 对应 lora_rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# --- 5. 准备数据集 ---
print(f"正在加载数据集: {dataset_path}")
# 假设是 jsonl 文件
dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

# --- 6. 配置训练参数 (TrainingArguments) ---
# 这里我们将您 YAML 文件中的配置翻译过来
training_args = TrainingArguments(
    output_dir="saves/qwen3-coder-30b/lora/mysql-sft-manual",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=500,
    
    bf16=True,  # 开启 bf16
    
    # --- 关键：DDP + MoE + 梯度检查点 的修复 ---
    gradient_checkpointing=True,          # 开启梯度检查点以节省显存
    ddp_find_unused_parameters=True,  # 必须！解决 MoE 在 DDP 下的梯度同步问题
    
    report_to="none", # 您可以改成 "tensorboard" 或 "wandb"
)

# --- 7. 初始化 SFTTrainer ---
# SFTTrainer 会自动处理数据打包和格式化（如果您提供了 formatting_func）
# 这里我们使用最简单的，假设 dataset_text_field 已经是格式化好的文本
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field=dataset_text_field, # 告知 SFTTrainer 使用哪个字段
    peft_config=lora_config,
    args=training_args,
    max_seq_length=4096,                   # 对应 cutoff_len
    packing=False,                         # 建议设为 True 以提高效率，但 False 更简单
)

# --- 8. 开始训练 ---
print("开始训练...")
trainer.train()

# --- 9. 保存模型 ---
print("训练完成，正在保存适配器...")
trainer.save_model(training_args.output_dir)
print(f"模型已保存至 {training_args.output_dir}")