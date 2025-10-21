# CUDA_VISIBLE_DEVICES=0 accelerate launch --num-processes 1 --config-file zero3.yaml rlvr.py
# CUDA_VISIBLE_DEVICES=1 vf-vllm --model willcb/Qwen3-8B --data-parallel-size 1 --enforce-eager --disable-log-requests --max-seq-len 10240 --host 127.0.0.1

import torch

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from verifiers import GRPOTrainer, GRPOConfig

import verifier as capa


OUTPUT_DIR = "data/rlvr_adapter"
MODEL_NAME = "willcb/Qwen3-8B"
HUB_MODEL_ID = "kyleavery/Qwen3-8B-EMBER2024-capa-rlvr"

set_seed(1337)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

vf_env = capa.load_environment(use_think=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules="all-linear",
)

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_generations=8,
    num_iterations=1,
    num_train_epochs=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    do_eval=False,
    eval_strategy="no",
    beta=0.0,
    top_p=0.95,
    temperature=1.0,
    weight_decay=0.0,
    max_grad_norm=1.0,
    mask_env_responses=True,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    save_strategy="steps",
    save_steps=15,
    save_only_model=True,
    save_total_limit=5,
    use_liger_kernel=True,
    bf16=True,
    fp16=False,
    dataloader_pin_memory=True,
    dataloader_drop_last=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    report_to=["wandb"],
    max_seq_len=16384,
    max_prompt_length=4096,
    seed=1337,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()
