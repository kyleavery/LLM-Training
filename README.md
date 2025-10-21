# LLM Training Example

Training an LLM on [joyce8/EMBER2024-capa](https://huggingface.co/datasets/joyce8/EMBER2024-capa) using SFT and RLVR.

## Supervised Fine-Tuning (SFT)

I used [create_sft_dataset.py](./create_sft_dataset.py) to generate [synthetic data](https://huggingface.co/datasets/kyleavery/EMBER2024-capa-cots) from [Qwen3-235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507). I trained [a model](https://huggingface.co/kyleavery/Qwen3-8B-EMBER2024-capa-sft) on this dataset using [sft.py](./sft.py).

## Reinforcement Learning with Verifiable Rewards (RLVR)

I trained [a model](https://huggingface.co/kyleavery/Qwen3-8B-EMBER2024-capa-rlvr) using the [capa verifier](./verifier.py) with [rlvr.py](./rlvr.py).

# Resources

- [Verifiers Repo](https://github.com/PrimeIntellect-ai/verifiers)
- [Hugging Face course: Supervised Fine-Tuning](https://huggingface.co/learn/llm-course/chapter11/1)
- [Hugging Face course: Build Reasoning Models](https://huggingface.co/learn/llm-course/chapter12/1)
- [Unsloth Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [Prime Intellect: How to use verifiers with Environments Hub](https://www.youtube.com/watch?v=04k8UsCYBvc)
- [Outflank blog: Training Specialist Models](https://www.outflank.nl/blog/2025/08/07/training-specialist-models/)
