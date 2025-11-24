#!/usr/bin/env python3
"""
数据预处理脚本:为JudgeBench数据生成评估重点关键词(key_focus_aspects)

JudgeBench数据特点:
1. 有两个jsonl文件(gpt-4o和claude),需要合并
2. 数据结构: pair_id, original_id, source, question, response_A, response_B, label
3. label格式: "A>B" 或 "B>A" 需要转换为chosen/rejected结构
4. 没有section字段,但有source字段表示数据来源

用法:
    # 测试模式
    python generate_key_aspects.py --model "gpt-4o-mini" --test_size 10

    # 全量处理
    python generate_key_aspects.py --model "gpt-4o-mini" --concurrency 15
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are an evaluation expert. Your task is to analyze a question and two candidate responses, then identify 3-5 key evaluation aspects that should be focused on when judging these responses.

Output ONLY a JSON array of strings, nothing else. Example: ["aspect1", "aspect2", "aspect3"]"""

USER_PROMPT_TEMPLATE = """This question is from the dataset: {source}

**Question:**
{prompt}

**Response A:**
{chosen}

**Response B:**
{rejected}

Observe the questions and paired answers that need to be evaluated. Then identify 3-5 key evaluation aspects for judging these responses. Output format: ["aspect1", "aspect2", "aspect3"]"""


# ============================================================================
# Core Functions
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], file_path: Path):
    """保存JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_judgebench_to_rewardbench_format(item: Dict) -> Dict:
    """
    将JudgeBench格式转换为RewardBench格式

    JudgeBench格式:
        {
            "pair_id": "...",
            "original_id": "...",
            "source": "mmlu-pro-law",
            "question": "...",
            "response_model": "gpt-4o-2024-05-13",
            "response_A": "...",
            "response_B": "...",
            "label": "A>B" or "B>A"
        }

    转换为RewardBench格式:
        {
            "id": "...",
            "prompt": "...",
            "chosen": "...",
            "rejected": "...",
            "source": "...",
            "response_model": "..."
        }
    """
    label = item["label"]

    # 根据label确定chosen和rejected
    if label == "A>B":
        chosen = item["response_A"]
        rejected = item["response_B"]
    elif label == "B>A":
        chosen = item["response_B"]
        rejected = item["response_A"]
    else:
        raise ValueError(f"Unknown label format: {label}")

    return {
        "id": item["pair_id"],
        "original_id": item.get("original_id"),
        "prompt": item["question"],
        "chosen": chosen,
        "rejected": rejected,
        "source": item["source"],
        "response_model": item["response_model"]
    }


def extract_json_array(text: str) -> Optional[List[str]]:
    """从文本中提取JSON数组"""
    try:
        result = json.loads(text.strip())
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except json.JSONDecodeError:
        pass

    # 正则提取兜底
    matches = re.findall(r'\[(.*?)\]', text, re.DOTALL)
    if matches:
        try:
            result = json.loads(f"[{matches[0]}]")
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return result
        except json.JSONDecodeError:
            pass

    return None


async def generate_key_aspects(
    client: AsyncOpenAI,
    model: str,
    item: Dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 10
) -> Dict:
    """为单条数据生成关键词"""
    source = item["source"]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        source=source,
        prompt=item["prompt"],
        chosen=item["chosen"],
        rejected=item["rejected"]
    )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )

                output = response.choices[0].message.content.strip()
                key_aspects = extract_json_array(output)

                if key_aspects:
                    item["key_focus_aspects"] = key_aspects[:5] if len(key_aspects) > 5 else key_aspects
                    return item

                if attempt == max_retries - 1:
                    item["key_focus_aspects"] = ["parsing_failed"]
                    item["_raw_output"] = output

            except Exception as e:
                if attempt == max_retries - 1:
                    item["key_focus_aspects"] = ["api_error"]
                    item["_error"] = str(e)
                else:
                    await asyncio.sleep(2 ** attempt)

    return item


async def process_all_data(
    client: AsyncOpenAI,
    model: str,
    all_data: List[Dict],
    concurrency: int
) -> List[Dict]:
    """并发处理所有数据"""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [generate_key_aspects(client, model, item, semaphore) for item in all_data]

    results = []
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
        result = await coro
        results.append(result)

    return results


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="为JudgeBench数据生成评估重点关键词")
    parser.add_argument("--api_base", type=str, default=None, help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--input_dir", type=str, default=".", help="输入目录")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument("--test_size", type=int, default=None, help="测试条数(不指定则全量处理)")

    args = parser.parse_args()

    # 初始化OpenAI客户端
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    api_base = args.api_base or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("必须提供 --api_key 或设置 OPENAI_API_KEY 环境变量")

    client_kwargs = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base

    client = AsyncOpenAI(**client_kwargs)

    # 读取JudgeBench的两个数据文件
    input_dir = Path(args.input_dir)
    judgebench_files = [
        "gpt-4o-2024-05-13.jsonl",
        "claude-3-5-sonnet-20240620.jsonl"
    ]

    all_data = []
    pair_id_to_item = {}  # 用于合并相同pair_id的数据

    for filename in judgebench_files:
        file_path = input_dir / filename
        if file_path.exists():
            raw_data = load_jsonl(file_path)
            print(f"Loaded {len(raw_data)} items from {filename}")

            # 转换格式并去重
            for raw_item in raw_data:
                pair_id = raw_item["pair_id"]
                # 如果已存在相同pair_id,跳过(保留第一个)
                if pair_id not in pair_id_to_item:
                    converted_item = convert_judgebench_to_rewardbench_format(raw_item)
                    pair_id_to_item[pair_id] = converted_item

    all_data = list(pair_id_to_item.values())
    print(f"Total unique pairs after merging: {len(all_data)}")

    if not all_data:
        print("No data to process")
        return

    # 测试模式
    if args.test_size is not None:
        all_data = all_data[:args.test_size]
        print(f"Test mode: processing {len(all_data)} items")
    else:
        print(f"Full mode: processing {len(all_data)} items")

    # 并发处理
    print(f"Model: {args.model}, Concurrency: {args.concurrency}\n")
    results = await process_all_data(client, args.model, all_data, args.concurrency)

    # 保存结果
    model_basename = os.path.basename(args.model)
    output_path = Path(args.output_dir) / f"judgebench_{model_basename}.jsonl"
    save_jsonl(results, output_path)

    # 统计
    failed = sum(1 for r in results if r.get("key_focus_aspects", [None])[0] in ["parsing_failed", "api_error"])
    print(f"\nCompleted: {output_path}")
    print(f"Success: {len(results) - failed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    # 按source统计
    source_counts = {}
    for r in results:
        source = r.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\nData distribution by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
