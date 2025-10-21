import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


MERGED_MODEL = "kyleavery/Qwen3-8B-EMBER2024-capa-sft-merged"
BASE_MODEL = "qwen/Qwen3-8B"
PEFT_ADAPTER = "kyleavery/Qwen3-8B-EMBER2024-capa-sft-adapter"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, PEFT_ADAPTER, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(PEFT_ADAPTER)

merged_model = model.merge_and_unload()

merged_model.push_to_hub(MERGED_MODEL, private=True)
tokenizer.push_to_hub(MERGED_MODEL, private=True)
