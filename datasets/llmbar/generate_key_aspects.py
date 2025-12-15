#!/usr/bin/env python3
"""
数据预处理脚本:为 LLMBar-Adversarial 每条数据生成评估重点关键词(key_focus_aspects)

LLMBar-Adversarial 是对抗性评估数据集，包含四个子集：
- Neighbor: 使用相近但不同的指令生成对抗输出
- GPTInst: GPT-4 生成的相似指令变体
- GPTOut: GPT-4 生成的表面优质但无用/错误的输出
- Manual: 人工构造的对抗样本

用法:
    # 测试模式 (每个 subset 取前5条)
    python generate_key_aspects.py --model "gpt-4o-mini" --test_size 5

    # 全量处理
    python generate_key_aspects.py --model "gpt-4o-mini" --concurrency 15
"""

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are an evaluation expert. Your task is to analyze a question and two candidate responses, then identify 3-5 key evaluation aspects that should be focused on when judging these responses.

Output ONLY a JSON array of strings, nothing else. Example: ["aspect1", "aspect2", "aspect3"]"""

SECTION_CONTEXT = {
    "neighbor": "This task evaluates instruction-following accuracy under adversarial conditions. ",
    "gptinst": "This task evaluates the ability to correctly interpret subtle instruction variants.",
    "gptout": "This task tests resistance to surface-quality deception.",
    "manual": "This task evaluates instruction-following under diverse adversarial strategies.",
}

# 控制是否在 prompt 中展示待评判的成对回复
INCLUDE_RESPONSES = False

USER_PROMPT_TEMPLATE_WITH_RESPONSES = """{section_context}

**Question:**
{prompt}

**Response A:**
{chosen}

**Response B:**
{rejected}

Observe the questions and paired answers that need to be evaluated. Then identify 3-5 key evaluation aspects for judging these responses. Output format: ["aspect1", "aspect2", "aspect3"]"""

USER_PROMPT_TEMPLATE_WITHOUT_RESPONSES = """{section_context}

**Question:**
{prompt}

Observe the question. Then identify 3-5 key evaluation aspects that should be focused on when judging responses to this question. Output format: ["aspect1", "aspect2", "aspect3"]"""


# ============================================================================
# Core Functions
# ============================================================================


def load_json(file_path: Path) -> List[Dict]:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], file_path: Path):
    """保存JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl_item(item: Dict, file_path: Path, lock: asyncio.Lock):
    """
    线程安全地追加单条数据到JSONL文件

    Args:
        item: 要保存的数据项
        file_path: 文件路径
        lock: asyncio锁，确保并发写入安全（在调用处使用）
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_existing_results(file_path: Path) -> Dict[str, Dict]:
    """
    加载已有结果文件，构建查找字典

    Returns:
        Dict[key, result]: key 由 prompt, chosen, rejected 三个字段生成
    """
    if not file_path.exists():
        return {}

    results_dict = {}
    try:
        data = load_jsonl(file_path)
        for item in data:
            # 生成唯一键（基于 prompt, chosen, rejected）
            key = _generate_item_key(item)
            results_dict[key] = item
    except Exception as e:
        print(f"Warning: Failed to load existing results from {file_path}: {e}")
        return {}

    return results_dict


def _generate_item_key(item: Dict) -> str:
    """
    根据 prompt, chosen, rejected 生成唯一键
    注意：LLMBar 原始数据使用 input, output_1, output_2
    """
    import hashlib

    # 获取 prompt（可能是 input 或 prompt）
    prompt = item.get("prompt", item.get("input", ""))

    # 对于已处理的数据，直接使用 chosen/rejected
    if "chosen" in item and "rejected" in item:
        chosen = item["chosen"]
        rejected = item["rejected"]
    # 对于原始数据，需要根据 label 确定 chosen/rejected
    elif "output_1" in item and "output_2" in item:
        output_1 = item["output_1"]
        output_2 = item["output_2"]
        label = item.get("label", 1)

        # LLMBar: label=1 表示 output_1 更好, label=2 表示 output_2 更好
        if label == 1:
            chosen = output_1
            rejected = output_2
        else:  # label == 2
            chosen = output_2
            rejected = output_1
    else:
        # 兜底情况
        chosen = ""
        rejected = ""

    key_fields = [prompt, chosen, rejected]
    # 使用 JSON 序列化确保一致性，然后计算哈希
    key_str = json.dumps(key_fields, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def filter_dataset_with_restore(
    dataset: List[Dict], existing_results: Dict[str, Dict]
) -> tuple[List[Dict], List[Dict]]:
    """
    根据已有结果过滤数据集

    Returns:
        (to_process, already_processed): 需要处理的数据和已处理的数据
    """
    to_process = []
    already_processed = []

    for item in dataset:
        key = _generate_item_key(item)
        if key in existing_results:
            # 使用已有结果
            already_processed.append(existing_results[key])
        else:
            # 需要重新处理
            to_process.append(item)

    return to_process, already_processed


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
    include_responses: bool = True,
    max_retries: int = 10,
) -> Dict:
    """为单条数据生成关键词"""
    subset = item["subset"]  # "neighbor", "gptinst", "gptout", "manual"
    section_context = SECTION_CONTEXT.get(subset, "This is a response evaluation task.")

    # 根据 INCLUDE_RESPONSES 选择合适的 prompt
    if include_responses:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_RESPONSES.format(
            section_context=section_context,
            prompt=item["prompt"],
            chosen=item["chosen"],
            rejected=item["rejected"],
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE_WITHOUT_RESPONSES.format(
            section_context=section_context, prompt=item["prompt"]
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
                    max_tokens=1024,
                )

                output = response.choices[0].message.content.strip()
                key_aspects = extract_json_array(output)

                if key_aspects:
                    item["key_focus_aspects"] = (
                        key_aspects[:5] if len(key_aspects) > 5 else key_aspects
                    )
                    return item

                # 解析失败，打印重试信息
                print(
                    f"⚠ Parsing failed for item subset={item.get('subset', 'unknown')} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(15)  # 等待15秒后重试
                else:
                    print(
                        f"❌ Parsing failed after {max_retries} attempts for item subset={item.get('subset', 'unknown')}"
                    )
                    item["key_focus_aspects"] = ["parsing_failed"]
                    item["_raw_output"] = output

            except Exception as e:
                print(
                    f"⚠ API error for item subset={item.get('subset', 'unknown')} (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(15)  # 等待15秒后重试
                else:
                    print(
                        f"❌ API error after {max_retries} attempts for item subset={item.get('subset', 'unknown')}"
                    )
                    item["key_focus_aspects"] = ["api_error"]
                    item["_error"] = str(e)

    return item


async def process_all_data(
    client: AsyncOpenAI,
    model: str,
    all_data: List[Dict],
    concurrency: int,
    output_path: Path,
    include_responses: bool = True,
) -> int:
    """
    并发处理所有数据，并增量保存到文件

    Returns:
        int: 成功处理的数据条数
    """
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()  # 文件写入锁

    tasks = [
        generate_key_aspects(client, model, item, semaphore, include_responses)
        for item in all_data
    ]

    processed_count = 0
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
        result = await coro

        # 每完成一条就立即保存
        async with file_lock:
            append_jsonl_item(result, output_path, file_lock)

        processed_count += 1

    return processed_count


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="为LLMBar数据生成评估重点关键词")
    parser.add_argument(
        "--api_base", type=str, default=None, help="OpenAI API base URL"
    )
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--input_dir", type=str, default=".", help="输入目录")
    parser.add_argument("--output_dir", type=str, default=".", help="输出目录")
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="每个 subset 采样的条数(不指定则全量处理)",
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

    # 读取 4 个子集的数据文件
    input_dir = Path(args.input_dir)
    subset_dirs = ["Neighbor", "GPTInst", "GPTOut", "Manual"]

    all_data = []
    idx = 0
    for subset_dir in subset_dirs:
        file_path = input_dir / subset_dir / "dataset.json"
        if file_path.exists():
            raw_data = load_json(file_path)
            # 添加 subset 和 id 字段，并转换为统一格式
            for item in raw_data:
                # 添加元信息
                item["subset"] = subset_dir.lower()
                item["id"] = idx
                idx += 1

                # 转换为统一格式: prompt, chosen, rejected
                item["prompt"] = item["input"]
                # LLMBar 数据集中 label=1 表示 output_1 更好, label=2 表示 output_2 更好
                if item.get("label", 1) == 1:
                    item["chosen"] = item["output_1"]
                    item["rejected"] = item["output_2"]
                else:  # label == 2
                    item["chosen"] = item["output_2"]
                    item["rejected"] = item["output_1"]

                # 删除已转换的原始字段
                for key in ["input", "output_1", "output_2", "label"]:
                    item.pop(key, None)

            all_data.extend(raw_data)
            print(f"Loaded {len(raw_data)} items from {subset_dir}")
        else:
            print(f"Warning: File not found: {file_path}")

    print(f"\nTotal: {len(all_data)} items from all subsets")

    if not all_data:
        print("No data to process")
        return

    # 统计subset分布
    subset_counts = {}
    for item in all_data:
        subset = item["subset"]
        subset_counts[subset] = subset_counts.get(subset, 0) + 1

    print("\nData distribution by subset:")
    for subset, count in sorted(subset_counts.items()):
        print(f"  {subset}: {count}")

    # 测试模式 - 按 subset 分层采样
    if args.test_size is not None:
        subset_data = defaultdict(list)
        for item in all_data:
            subset_data[item["subset"]].append(item)

        sampled_data = []
        for subset in sorted(subset_data.keys()):
            sampled = subset_data[subset][: args.test_size]
            sampled_data.extend(sampled)
            print(f"Sampled {len(sampled)} items from subset: {subset}")

        all_data = sampled_data
        print(f"\nTest mode: processing {len(all_data)} items total")
    else:
        print(f"\nFull mode: processing {len(all_data)} items")

    # Restore模式：加载已有结果并过滤
    model_basename = os.path.basename(args.model)
    output_path = Path(args.output_dir) / f"llmbar_key_{model_basename}.jsonl"

    existing_results = load_existing_results(output_path)
    already_processed_count = len(existing_results)

    if existing_results:
        all_data, already_processed_list = filter_dataset_with_restore(
            all_data, existing_results
        )
        print(f"\n✓ Restore mode enabled:")
        print(f"  - Found {already_processed_count} existing results")
        print(f"  - Need to process {len(all_data)} new items")

        # 将已有结果写入输出文件（覆盖模式，确保文件从干净状态开始）
        save_jsonl(already_processed_list, output_path)
    else:
        print(f"\n✓ No existing results found, processing all {len(all_data)} items")
        # 确保输出文件为空（如果存在的话）
        if output_path.exists():
            output_path.unlink()

    # 并发处理（新数据会增量追加到output_path）
    print(f"\nModel: {args.model}, Concurrency: {args.concurrency}")
    print(f"Include responses in prompt: {INCLUDE_RESPONSES}\n")
    newly_processed_count = await process_all_data(
        client, args.model, all_data, args.concurrency, output_path, INCLUDE_RESPONSES
    )

    # 统计（重新读取完整文件）
    final_results = load_jsonl(output_path)
    failed = sum(
        1
        for r in final_results
        if r.get("key_focus_aspects", [None])[0] in ["parsing_failed", "api_error"]
    )

    print(f"\n✓ Incremental save completed:")
    print(f"  - Previously processed: {already_processed_count}")
    print(f"  - Newly processed: {newly_processed_count}")
    print(f"  - Total in file: {len(final_results)}")
    print(f"\nCompleted: {output_path}")
    print(f"Success: {len(final_results) - failed}/{len(final_results)}")
    print(f"Failed: {failed}/{len(final_results)}")


if __name__ == "__main__":
    asyncio.run(main())
