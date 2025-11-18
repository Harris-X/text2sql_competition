#!/usr/bin/env python
import json
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(sample: dict) -> str:
    instruction = sample.get("instruction", "").strip()
    input_text = sample.get("input", "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    if instruction:
        return instruction
    return input_text


def main():
    base_model_path = Path("/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct").resolve()
    adapter_path = Path("/root/autodl-tmp/comp/LLaMA-Factory/saves/qwen3-coder-30b/lora/bird_sft").resolve()
    data_path = Path("/root/autodl-tmp/comp/LLaMA-Factory/data/bird_mini_text2sql_alpaca.json").resolve()
    output_path = Path("/root/autodl-tmp/comp/LLaMA-Factory/data/bird_mini_text2sql_dpo.jsonl").resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to(device)
    model.eval()

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gen_cfg = dict(max_new_tokens=256, temperature=0.8, top_p=0.9, do_sample=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(data):
            prompt = build_prompt(sample)
            messages = [{"role": "user", "content": prompt}]
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = (model_inputs != tokenizer.pad_token_id).long()

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=model_inputs,
                    attention_mask=attention_mask,
                    **gen_cfg,
                )
            gen_tokens = output_ids[0, model_inputs.shape[1]:]
            rejected = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            chosen = sample.get("output", "").strip()
            record = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected or "SELECT 1;",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx < 3:
                print("==== SAMPLE", idx, "====")
                print("PROMPT:\n", prompt[:400])
                print("CHOSEN:\n", chosen)
                print("REJECTED:\n", rejected)

    print("DPO data saved to", output_path)


if __name__ == "__main__":
    main()
