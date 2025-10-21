import datasets
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig


OUTPUT_DIR = "data/sft_adapter"
MODEL_NAME = "willcb/Qwen3-8B"
DATASET_NAME = "kyleavery/EMBER2024-capa-cots"
HUB_MODEL_ID = "kyleavery/Qwen3-8B-EMBER2024-capa-sft"

set_seed(1337)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

train_dataset = datasets.load_dataset(DATASET_NAME, split="train")

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

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    assistant_only_loss=True,
    chat_template_path="chat_template.jinja",
    do_eval=False,
    eval_strategy="no",
    weight_decay=0.001,
    warmup_ratio=0.05,
    max_grad_norm=0.2,
    logging_steps=1,
    save_strategy="epoch",
    save_only_model=True,
    save_total_limit=1,
    use_liger_kernel=True,
    bf16=True,
    fp16=False,
    dataloader_pin_memory=True,
    dataloader_drop_last=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to=["wandb"],
    optim="adamw_torch",
    max_length=32768,
    packing=False,
    padding_free=True,
    seed=1337,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()
