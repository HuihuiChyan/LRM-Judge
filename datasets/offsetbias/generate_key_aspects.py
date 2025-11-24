#!/usr/bin/env python3
"""
OffsetBias数据集预处理脚本:为每条数据生成评估重点关键词(key_focus_aspects)

OffsetBias数据特点:
1. JSON格式(非JSONL),包含多个bias类型
2. 数据结构: instruction, response1, response2, label, bias
3. label=1表示response2更好(chosen), label=0表示response1更好
4. bias字段表示数据的偏差类型

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

USER_PROMPT_TEMPLATE = """This is a bias detection task. The bias type being tested is: {bias_type}

**Instruction:**
{prompt}

**Response A:**
{chosen}

**Response B:**
{rejected}

Observe the instruction and paired responses that need to be evaluated. Then identify 3-5 key evaluation aspects for judging these responses, considering the bias type being tested. Output format: ["aspect1", "aspect2", "aspect3"]"""


# ============================================================================
# Core Functions
# ============================================================================


def load_offsetbias_data(file_path: Path) -> List[Dict]:
    """加载OffsetBias JSON文件并转换为列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_items = []
    item_id = 0

    # 遍历所有bias类型
    for _, items in data.items():
        for item in items:
            item_id += 1
            all_items.append(
                {
                    "id": f"offsetbias_{item_id}",
                    "bias": item["bias"],
                    "instruction": item["instruction"],
                    "response1": item["response1"],
                    "response2": item["response2"],
                    "label": item["label"],
                }
            )

    return all_items


def convert_offsetbias_to_rewardbench_format(item: Dict) -> Dict:
    """
    将OffsetBias格式转换为RewardBench格式

    OffsetBias格式:
        {
            "id": "offsetbias_1",
            "bias": "length bias",
            "instruction": "...",
            "response1": "...",
            "response2": "...",
            "label": 1  # label=1表示response1更好(所有数据都是label=1)
        }

    转换为RewardBench格式:
        {
            "id": "offsetbias_1",
            "prompt": "...",
            "chosen": "...",
            "rejected": "...",
            "bias": "...",
        }
    """
    # OffsetBias数据集中label=1表示response1更好
    # 所有数据都是label=1
    chosen = item["response1"]
    rejected = item["response2"]

    return {
        "id": item["id"],
        "prompt": item["instruction"],
        "chosen": chosen,
        "rejected": rejected,
        "bias": item["bias"],
    }


def save_jsonl(data: List[Dict], file_path: Path):
    """保存JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_json_array(text: str) -> Optional[List[str]]:
    """从文本中提取JSON数组"""
    try:
        result = json.loads(text.strip())
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except json.JSONDecodeError:
        pass

    # 正则提取兜底
    matches = re.findall(r"\[(.*?)\]", text, re.DOTALL)
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
    max_retries: int = 10,
) -> Dict:
    """为单条数据生成关键词"""
    bias_type = item["bias"]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        bias_type=bias_type,
        prompt=item["prompt"],
        chosen=item["chosen"],
        rejected=item["rejected"],
    )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )

                output = response.choices[0].message.content.strip()
                key_aspects = extract_json_array(output)

                if key_aspects:
                    item["key_focus_aspects"] = (
                        key_aspects[:5] if len(key_aspects) > 5 else key_aspects
                    )
                    return item

                if attempt == max_retries - 1:
                    item["key_focus_aspects"] = ["parsing_failed"]
                    item["_raw_output"] = output

            except Exception as e:
                if attempt == max_retries - 1:
                    item["key_focus_aspects"] = ["api_error"]
                    item["_error"] = str(e)
                else:
                    await asyncio.sleep(2**attempt)

    return item


async def process_all_data(
    client: AsyncOpenAI, model: str, all_data: List[Dict], concurrency: int
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
    parser = argparse.ArgumentParser(description="为OffsetBias数据生成评估重点关键词")
    parser.add_argument(
        "--api_base", type=str, default=None, help="OpenAI API base URL"
    )
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument(
        "--input_file", type=str, default="biasbench.json", help="输入文件"
    )
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument(
        "--test_size", type=int, default=None, help="测试条数(不指定则全量处理)"
    )

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

    # 读取OffsetBias数据
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_data = load_offsetbias_data(input_path)
    print(f"Loaded {len(raw_data)} items from {input_path.name}")

    # 转换为RewardBench格式
    all_data = [convert_offsetbias_to_rewardbench_format(item) for item in raw_data]

    # 统计bias类型分布
    bias_counts = {}
    for item in all_data:
        bias = item["bias"]
        bias_counts[bias] = bias_counts.get(bias, 0) + 1

    print("\nData distribution by bias type:")
    for bias, count in sorted(bias_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {bias}: {count}")

    # 测试模式
    if args.test_size is not None:
        all_data = all_data[: args.test_size]
        print(f"\nTest mode: processing {len(all_data)} items")
    else:
        print(f"\nFull mode: processing {len(all_data)} items")

    # 并发处理
    print(f"Model: {args.model}, Concurrency: {args.concurrency}\n")
    results = await process_all_data(client, args.model, all_data, args.concurrency)

    # 保存结果
    model_basename = os.path.basename(args.model)
    output_path = Path(args.output_dir) / f"offsetbias_{model_basename}.jsonl"
    save_jsonl(results, output_path)

    # 统计
    failed = sum(
        1
        for r in results
        if r.get("key_focus_aspects", [None])[0] in ["parsing_failed", "api_error"]
    )
    print(f"\nCompleted: {output_path}")
    print(f"Success: {len(results) - failed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
