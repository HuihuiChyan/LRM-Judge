#!/usr/bin/env python3
"""
数据预处理脚本:为 OffsetBias 每条数据生成评估计划(evaluation_plan)

用法:
    # 测试模式 (每个 bias_type 取前10条)
    python generate_evaluation_plan.py --model "gpt-4o-mini" --test_size 10

    # 全量处理
    python generate_evaluation_plan.py --model "gpt-4o-mini" --concurrency 15
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

SECTION_CONTEXT = {
    "length bias": "This task tests whether the judge incorrectly prefers longer responses over concise, accurate ones. The correct response is typically brief and directly addresses the question, while the inferior response is unnecessarily verbose or contains irrelevant elaboration. Focus on accuracy and relevance, not length.",
    "concreteness": "This task tests whether the judge incorrectly prefers responses with specific details/citations over those that directly answer the question. The correct response addresses the actual question clearly, while the inferior response may include concrete references but is off-topic or factually wrong. Prioritize relevance and correctness over superficial concreteness.",
    "empty reference": "This task tests whether the judge incorrectly prefers fabricated content when the user's instruction lacks context. The correct response should ask for clarification when information is missing, while the inferior response fabricates or assumes missing details. Reward appropriate uncertainty handling.",
    "content_continuation": "This task tests whether the judge incorrectly prefers responses that continue or elaborate beyond instructions. The correct response strictly follows the given instruction (e.g., rewrite, paraphrase, correct), while the inferior response adds unsolicited continuation or expansion. Prioritize instruction adherence over perceived fluency.",
    "nested_instruction": "This task tests whether the judge correctly interprets meta-level instructions (summarize, rephrase, improve). The correct response addresses the meta-instruction (e.g., summarizing a passage), while the inferior response misinterprets it as a direct question and answers the content itself. Focus on understanding the true task intent.",
    "familiar knowledge preference bias": "This task tests whether the judge incorrectly prefers familiar but irrelevant information. The correct response answers the actual question asked, while the inferior response provides related but off-topic knowledge that seems more familiar. Prioritize relevance to the specific question.",
}

# 用户可以通过注释/取消注释以下两个 prompt 来控制是否给模型展示待评判的成对回复
# INCLUDE_RESPONSES = True 会展示回复内容，False 则不展示
INCLUDE_RESPONSES = False

USER_PROMPT_TEMPLATE_WITH_RESPONSES = """We want to evaluate the quality of the responses provided by AI assistants to the user question displayed below. For that, your task is to help us build an evaluation plan that can then be executed to assess the response quality. Whenever appropriate, you can choose to also include a step-by-step reference answer as part of the evaluation plan. Enclose your evaluation plan between the tags "[Start of Evaluation Plan]" and "[End of Evaluation Plan]".

Evaluation Domain: {section_context}

[User Question]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}"""

USER_PROMPT_TEMPLATE_WITHOUT_RESPONSES = """We want to evaluate the quality of the responses provided by AI assistants to the user question displayed below. For that, your task is to help us build an evaluation plan that can then be executed to assess the response quality. Whenever appropriate, you can choose to also include a step-by-step reference answer as part of the evaluation plan. Enclose your evaluation plan between the tags "[Start of Evaluation Plan]" and "[End of Evaluation Plan]".

Evaluation Domain: {section_context}

[User Question]
{instruction}"""


# ============================================================================
# Core Functions
# ============================================================================


def load_json(file_path: Path) -> Dict:
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        lock: asyncio锁，确保并发写入安全
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")



def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    注意：OffsetBias 原始数据使用 instruction, response1, response2
    """
    import hashlib

    # 获取 prompt（可能是 instruction 或 prompt）
    prompt = item.get("prompt", item.get("instruction", ""))

    # 对于已处理的数据，直接使用 chosen/rejected
    if "chosen" in item and "rejected" in item:
        chosen = item["chosen"]
        rejected = item["rejected"]
    # 对于原始数据，需要根据 label 确定 chosen/rejected
    elif "response1" in item and "response2" in item:
        response1 = item["response1"]
        response2 = item["response2"]
        label = item.get("label", 1)

        # OffsetBias: label=1 表示 response1 更好
        if label == 1:
            chosen = response1
            rejected = response2
        else:
            chosen = response2
            rejected = response1
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


def extract_evaluation_plan(text: str) -> Optional[str]:
    """从文本中提取 evaluation plan"""
    # 提取 [Start of Evaluation Plan] 和 [End of Evaluation Plan] 之间的内容
    pattern = r"\[Start of Evaluation Plan\](.*?)\[End of Evaluation Plan\]"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        result = match.group(1).strip()
    else:
        # 如果没有找到标记，返回整个文本作为 evaluation plan
        result = text.strip()

    # 剔除 <think></think> 标签及其内容（用于 DeepSeek-R1 等推理模型）
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL | re.IGNORECASE)

    return result.strip()


async def generate_evaluation_plan(
    client: AsyncOpenAI,
    model: str,
    item: Dict,
    semaphore: asyncio.Semaphore,
    include_responses: bool = True,
    max_retries: int = 10,
) -> Dict:
    """为单条数据生成评估计划"""
    bias_type = item["bias"]
    section_context = SECTION_CONTEXT.get(
        bias_type, "This is a response evaluation task."
    )

    # 根据 INCLUDE_RESPONSES 选择合适的 prompt
    if include_responses:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_RESPONSES.format(
            section_context=section_context,
            instruction=item["instruction"],
            response_a=item["response1"],
            response_b=item["response2"],
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE_WITHOUT_RESPONSES.format(
            section_context=section_context, instruction=item["instruction"]
        )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7,
                    max_tokens=8192,
                )

                output = response.choices[0].message.content.strip()
                evaluation_plan = extract_evaluation_plan(output)

                if evaluation_plan:
                    item["evaluation_plan"] = evaluation_plan
                    # 转换为统一格式: prompt, chosen, rejected
                    item["prompt"] = item["instruction"]
                    # OffsetBias 数据集中 label=1 表示 response1 更好
                    # 根据原 generate_key_aspects.py 注释，所有数据都是 label=1
                    if item.get("label", 1) == 1:
                        item["chosen"] = item["response1"]
                        item["rejected"] = item["response2"]
                    else:
                        item["chosen"] = item["response2"]
                        item["rejected"] = item["response1"]

                    # 删除已转换的原始字段，只保留必要字段
                    for key in ["instruction", "response1", "response2", "label"]:
                        item.pop(key, None)

                    return item

                # 解析失败，打印重试信息
                print(f"⚠ Parsing failed for item bias={item.get('bias', 'unknown')} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(15)  # 等待15秒后重试
                else:
                    print(f"❌ Parsing failed after {max_retries} attempts for item bias={item.get('bias', 'unknown')}")
                    item["evaluation_plan"] = "parsing_failed"
                    item["_raw_output"] = output

            except Exception as e:
                print(f"⚠ API error for item bias={item.get('bias', 'unknown')} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(15)  # 等待15秒后重试
                else:
                    print(f"❌ API error after {max_retries} attempts for item bias={item.get('bias', 'unknown')}")
                    item["evaluation_plan"] = "api_error"
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
        generate_evaluation_plan(client, model, item, semaphore, include_responses)
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
    parser = argparse.ArgumentParser(description="生成评估计划 (OffsetBias)")
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
        help="每个 bias_type 采样的条数(不指定则全量处理)",
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

    # 读取数据文件
    input_dir = Path(args.input_dir)
    input_file = input_dir / "biasbench.json"

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    # 加载并展平数据
    raw_data = load_json(input_file)
    all_data = []
    idx = 0
    for bias_type, items in raw_data.items():
        for item in items:
            item["id"] = idx
            all_data.append(item)
            idx += 1

    print(f"Loaded {len(all_data)} items from {input_file.name}")

    if not all_data:
        print("No data to process")
        return

    # 测试模式 - 按 bias_type 分层采样
    if args.test_size is not None:
        # 统计每个 bias_type 的数据量
        bias_data = defaultdict(list)
        for item in all_data:
            bias_data[item["bias"]].append(item)

        # 每个 bias_type 取前 N 条
        sampled_data = []
        for bias_type in sorted(bias_data.keys()):
            sampled = bias_data[bias_type][: args.test_size]
            sampled_data.extend(sampled)
            print(f"Sampled {len(sampled)} items from bias_type: {bias_type}")

        all_data = sampled_data
        print(f"Test mode: processing {len(all_data)} items total")
    else:
        print(f"Full mode: processing {len(all_data)} items")

    # Restore模式：加载已有结果并过滤
    model_basename = os.path.basename(args.model)
    output_path = Path(args.output_dir) / f"offsetbias_{model_basename}.jsonl"

    existing_results = load_existing_results(output_path)
    already_processed_count = len(existing_results)

    if existing_results:
        all_data, already_processed_list = filter_dataset_with_restore(all_data, existing_results)
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
        if r.get("evaluation_plan") in ["parsing_failed", "api_error"]
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
