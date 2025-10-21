import json
from typing import List, Dict, Any

import asyncio
import aiofiles

from asyncio import Queue
from datasets import load_dataset
from openai import AsyncOpenAI

import verifier as capa


OUTPUT_FILE = "responses.jsonl"
MODEL_NAME = "qwen/qwen3-235b-a22b-thinking-2507"
DATASET_NAME = "kyleavery/EMBER2024-capa-cots"
NUM_WORKERS = 8
OPENROUTER_API_KEY = ""


openrouter_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
)


async def generate_and_score_worker(
    queue: Queue,
    worker_id: int,
    items: List[Dict[str, Any]],
    system_prompt: str,
    parser: Any,
    rubric: Any,
) -> None:
    for row in items:
        question: str = row["question"]
        gold_answer: str = row["answer"]

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            response = await openrouter_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                extra_body={"reasoning": {"enabled": True}},
            )
        except Exception as e:
            print(f"Worker {worker_id}: Error from OpenRouter: {e}")
            continue

        try:
            try:
                reasoning = response.choices[0].message.reasoning or ""
            except Exception:
                reasoning = ""
            content = response.choices[0].message.content or ""
            output = f"<think>\n{reasoning}</think>\n\n{content}".strip()
        except Exception as e:
            print(f"Worker {worker_id}: Error processing response: {e}")
            continue

        completion_messages = messages + [{"role": "assistant", "content": output}]

        try:
            reward = float(
                rubric.correct_features_reward_func(
                    parser=parser, completion=completion_messages, answer=gold_answer
                )
            )
        except Exception as e:
            print(f"Worker {worker_id}: Error computing reward: {e}")
            reward = 0.0

        if reward < 1.0:
            continue

        try:
            result = json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": output},
                    ],
                    "answer": gold_answer,
                }
            )
            await queue.put(result)
        except Exception as e:
            print(f"Worker {worker_id}: Error queueing result: {e}")
            continue


async def writer(queue: Queue, output_file: str) -> None:
    while True:
        result = await queue.get()
        try:
            if result is None:
                queue.task_done()
                break
            async with aiofiles.open(output_file, "a", encoding="utf-8") as f:
                await f.write(result + "\n")
        except Exception as e:
            print("Writer: Error writing to file:", e)
        finally:
            if result is not None:
                queue.task_done()


async def main():
    env = capa.load_environment(
        use_think=True, feature_mode="id", seed=1337, split="train", n=10_000
    )
    system_prompt = getattr(env, "system_prompt", None)
    parser = getattr(env, "parser", None)
    rubric = getattr(env, "rubric", None)
    dataset = getattr(env, "dataset", None)

    if dataset is None or rubric is None or parser is None or system_prompt is None:
        raise RuntimeError("Failed to load verifier environment")

    items: List[Dict[str, Any]] = []
    for rec in dataset:
        q = rec.get("question")
        a = rec.get("answer")
        if not isinstance(q, str):
            continue
        items.append({"question": q, "answer": a})

    queue = Queue()
    writer_task = asyncio.create_task(writer(queue, OUTPUT_FILE))

    workers = []
    num_workers = min(NUM_WORKERS, max(1, len(items)))
    total = len(items)

    base = total // num_workers
    extra = total % num_workers
    start = 0
    for i in range(num_workers):
        end = start + base + (1 if i < extra else 0)
        shard = items[start:end]
        worker = asyncio.create_task(
            generate_and_score_worker(queue, i, shard, system_prompt, parser, rubric)
        )
        workers.append(worker)
        start = end

    await asyncio.gather(*workers)

    await queue.put(None)
    await writer_task

    dataset = load_dataset("json", data_files="responses.jsonl")
    dataset.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
