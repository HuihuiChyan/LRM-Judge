#!/usr/bin/env python3
"""
数据预处理脚本:为 JudgeBench 每条数据生成评估计划(evaluation_plan)

用法:
    # 测试模式 (每个 source 取前10条)
    python generate_evaluation_plan.py --model "gpt-4o-mini" --test_size 10

    # 全量处理
    python generate_evaluation_plan.py --model "gpt-4o-mini" --concurrency 15
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm


# ============================================================================
# Prompt Templates
# ============================================================================

SECTION_CONTEXT = {
    "mmlu-pro": "This is a multiple-choice question from MMLU-Pro dataset.",
    "livebench-reasoning": "This is a reasoning task from LiveBench.",
    "livebench-math": "This is a math problem from LiveBench.",
    "livecodebench": "This is a coding problem from LiveCodeBench.",
}

# 用户可以通过注释/取消注释以下两个 prompt 来控制是否给模型展示待评判的成对回复
# INCLUDE_RESPONSES = True 会展示回复内容，False 则不展示
INCLUDE_RESPONSES = True

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


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], file_path: Path):
    """保存JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_evaluation_plan(text: str) -> Optional[str]:
    """从文本中提取 evaluation plan"""
    # 提取 [Start of Evaluation Plan] 和 [End of Evaluation Plan] 之间的内容
    pattern = r"\[Start of Evaluation Plan\](.*?)\[End of Evaluation Plan\]"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    # 如果没有找到标记，返回整个文本作为 evaluation plan
    return text.strip()


def get_section_context(source: str) -> str:
    """根据 source 获取对应的 section context"""
    # 判断 source 属于哪个大类
    if source.startswith("mmlu-pro"):
        return SECTION_CONTEXT["mmlu-pro"]
    elif source == "livebench-reasoning":
        return SECTION_CONTEXT["livebench-reasoning"]
    elif source == "livebench-math":
        return SECTION_CONTEXT["livebench-math"]
    elif source == "livecodebench":
        return SECTION_CONTEXT["livecodebench"]
    else:
        return "This is a question-answering task."


async def generate_evaluation_plan(
    client: AsyncOpenAI,
    model: str,
    item: Dict,
    semaphore: asyncio.Semaphore,
    include_responses: bool = True,
    max_retries: int = 10,
) -> Dict:
    """为单条数据生成评估计划"""
    source = item["source"]
    section_context = get_section_context(source)

    # 根据 INCLUDE_RESPONSES 选择合适的 prompt
    if include_responses:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_RESPONSES.format(
            section_context=section_context,
            instruction=item["question"],
            response_a=item["response_A"],
            response_b=item["response_B"],
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE_WITHOUT_RESPONSES.format(
            section_context=section_context, instruction=item["question"]
        )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7,
                    max_tokens=4096,
                )

                output = response.choices[0].message.content.strip()
                evaluation_plan = extract_evaluation_plan(output)

                if evaluation_plan:
                    item["evaluation_plan"] = evaluation_plan
                    # 转换为统一格式: prompt, chosen, rejected
                    item["prompt"] = item["question"]
                    # JudgeBench 的 label 格式是 "A>B" 或 "B>A"
                    if item.get("label") == "A>B":
                        item["chosen"] = item["response_A"]
                        item["rejected"] = item["response_B"]
                    else:  # "B>A"
                        item["chosen"] = item["response_B"]
                        item["rejected"] = item["response_A"]
                    return item

                if attempt == max_retries - 1:
                    item["evaluation_plan"] = "parsing_failed"
                    item["_raw_output"] = output

            except Exception as e:
                if attempt == max_retries - 1:
                    item["evaluation_plan"] = "api_error"
                    item["_error"] = str(e)
                else:
                    await asyncio.sleep(2**attempt)

    return item


async def process_all_data(
    client: AsyncOpenAI,
    model: str,
    all_data: List[Dict],
    concurrency: int,
    include_responses: bool = True,
) -> List[Dict]:
    """并发处理所有数据"""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        generate_evaluation_plan(client, model, item, semaphore, include_responses)
        for item in all_data
    ]

    results = []
    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
        result = await coro
        results.append(result)

    return results


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="生成评估计划 (JudgeBench)")
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
        help="每个 source 采样的条数(不指定则全量处理)",
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

    # 读取数据文件 (支持两个文件)
    input_dir = Path(args.input_dir)
    input_files = ["gpt-4o-2024-05-13.jsonl", "claude-3-5-sonnet-20240620.jsonl"]

    all_data = []
    for filename in input_files:
        file_path = input_dir / filename
        if file_path.exists():
            data = load_jsonl(file_path)
            all_data.extend(data)
            print(f"Loaded {len(data)} items from {filename}")

    if not all_data:
        print("No data to process")
        return

    # 测试模式 - 按 source 分层采样
    if args.test_size is not None:
        # 统计每个 source 的数据量
        source_data = defaultdict(list)
        for item in all_data:
            source_data[item["source"]].append(item)

        # 每个 source 取前 N 条
        sampled_data = []
        for source in sorted(source_data.keys()):
            sampled = source_data[source][: args.test_size]
            sampled_data.extend(sampled)
            print(f"Sampled {len(sampled)} items from source: {source}")

        all_data = sampled_data
        print(f"Test mode: processing {len(all_data)} items total")
    else:
        print(f"Full mode: processing {len(all_data)} items")

    # 并发处理
    print(f"Model: {args.model}, Concurrency: {args.concurrency}")
    print(f"Include responses in prompt: {INCLUDE_RESPONSES}\n")
    results = await process_all_data(
        client, args.model, all_data, args.concurrency, INCLUDE_RESPONSES
    )

    # 保存结果
    model_basename = os.path.basename(args.model)
    output_path = Path(args.output_dir) / f"judgebench_{model_basename}.jsonl"
    save_jsonl(results, output_path)

    # 统计
    failed = sum(
        1
        for r in results
        if r.get("evaluation_plan") in ["parsing_failed", "api_error"]
    )
    print(f"\nCompleted: {output_path}")
    print(f"Success: {len(results) - failed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
