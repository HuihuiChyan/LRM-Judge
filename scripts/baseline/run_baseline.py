#!/usr/bin/env python3
"""
Baseline Judge Model Evaluation Script
使用原版 RewardBench prompt 进行评估作为对照组

用法:
    # 普通模型
    python run_baseline.py --model "gpt-4o" --api_key "xxx" --api_base "https://api.openai.com/v1" --data_file "datasets/rewardbench/Chat.jsonl" --concurrency 10

    # 推理模型
    python run_baseline.py --model "deepseek-reasoner" --api_key "xxx" --api_base "https://api.deepseek.com" --data_file "datasets/rewardbench/Chat.jsonl" --reasoning_model --concurrency 5
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
from openai import OpenAI
from tqdm import tqdm

# 从 self_synthesized 导入相同的配置
import sys

sys.path.append(str(Path(__file__).parent.parent / "self_synthesized"))
from run_self_synthesized import (
    API_MAX_RETRY,
    API_RETRY_SLEEP,
    API_ERROR_OUTPUT,
    chat_completion_openai,
    extract_reasoning,
    load_jsonl,
    save_jsonl,
    load_existing_results,
    _generate_item_key,
    filter_dataset_with_restore,
)

# ============================================================================
# Prompt Templates (原版 RewardBench Prompt)
# ============================================================================

SYSTEM_PROMPT = """You are an impartial judge evaluating AI assistant responses."""

USER_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[[User Question]]
{prompt}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]"""


# ============================================================================
# Core Functions
# ============================================================================


def process_judgement(output: str) -> Tuple[Optional[str], str]:
    """
    从模型输出中提取评判结果

    Returns:
        (verdict, raw_output)
        verdict: "A" 表示 A 更好, "B" 表示 B 更好, None 表示解析失败
    """
    # 提取最后的判决标记
    matches = re.findall(r"\[\[([AB])\]\]", output)
    if not matches:
        return None, output

    verdict = matches[-1]  # 取最后一个匹配

    if verdict == "A":
        return "A", output
    elif verdict == "B":
        return "B", output
    else:
        return None, output


def get_judgement(
    client: OpenAI,
    model: str,
    item: Dict,
    max_tokens: int = 16384,
    is_reasoning_model: bool = False,
    test_mode: bool = False,
) -> Dict:
    """
    对单个数据项进行评判

    Args:
        client: OpenAI 客户端
        model: 模型名称
        item: 数据项,包含 prompt, chosen, rejected 等字段
        max_tokens: 最大生成 token 数
        is_reasoning_model: 是否为推理模型(会自动添加 thinking_budget=16384)
        test_mode: 测试模式,如果为 True 会输出每条数据的模型输出

    Returns:
        包含评判结果的字典
    """
    # 随机打乱答案顺序以减少位置偏差
    shuffle = random.choice([True, False])
    if shuffle:
        response_a = item["rejected"]
        response_b = item["chosen"]
    else:
        response_a = item["chosen"]
        response_b = item["rejected"]

    # 构建提示
    user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt=item["prompt"],
        response_a=response_a,
        response_b=response_b,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 调用模型
    output, reasoning = chat_completion_openai(
        client=client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        is_reasoning_model=is_reasoning_model,
    )

    # 测试模式下输出模型响应
    if test_mode:
        print("\n" + "=" * 80)
        print(
            f"📝 Item ID: {item.get('id', 'N/A')} | Section: {item.get('section', 'N/A')}"
        )
        print("=" * 80)
        print("**模型输出:**")
        print(output)
        print("=" * 80 + "\n")

    # 处理判决结果
    verdict, raw_output = process_judgement(output)

    # 根据 shuffle 转换判决结果为最终得分
    score = None
    if verdict == "A":
        score = 0 if shuffle else 1  # A 是 rejected 则为 0,否则为 1
    elif verdict == "B":
        score = 1 if shuffle else 0  # B 是 chosen 则为 1,否则为 0

    # 构建返回结果
    result = {
        **item,  # 保留原始数据
        "model_output": raw_output,
        "score": score,
        "shuffle": shuffle,
    }

    # 如果有推理内容,单独提取
    if reasoning:
        result["reasoning"] = reasoning

    return result


def evaluate_dataset(
    client: OpenAI,
    model: str,
    dataset: List[Dict],
    max_tokens: int = 16384,
    is_reasoning_model: bool = False,
    test_mode: bool = False,
    num_threads: int = 10,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    并发评估整个数据集

    Args:
        client: OpenAI 客户端
        model: 模型名称
        dataset: 数据集
        max_tokens: 最大生成 token 数
        is_reasoning_model: 是否为推理模型
        test_mode: 测试模式,会输出每条数据的模型输出
        num_threads: 并发线程数
        output_path: 输出文件路径(用于增量保存)

    Returns:
        评估结果列表
    """
    results = [None] * len(dataset)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {
            executor.submit(
                get_judgement,
                client,
                model,
                item,
                max_tokens,
                is_reasoning_model,
                test_mode,
            ): i
            for i, item in enumerate(dataset)
        }

        # 使用 tqdm 显示进度
        with tqdm(total=len(dataset), desc="Evaluating") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result

                    # 增量保存结果
                    if output_path and result:
                        save_jsonl([r for r in results if r is not None], output_path)

                except Exception as e:
                    print(f"\n❌ Error processing item {index}: {e}")
                    results[index] = {
                        **dataset[index],
                        "model_output": f"Error: {str(e)}",
                        "score": None,
                        "error": str(e),
                    }

                pbar.update(1)

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Judge Model Evaluation (原版 RewardBench Prompt)"
    )

    # 模型配置
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--api_base", type=str, required=True, help="API Base URL")
    parser.add_argument(
        "--max_tokens", type=int, default=16384, help="最大生成 token 数"
    )

    # 推理模型配置
    parser.add_argument(
        "--reasoning_model",
        action="store_true",
        help="是否为推理模型(自动启用 thinking_budget=16384)",
    )

    # 数据配置
    parser.add_argument(
        "--data_file", type=str, required=True, help="输入数据文件路径 (jsonl)"
    )

    # 执行配置
    parser.add_argument("--concurrency", type=int, default=10, help="并发线程数")
    parser.add_argument(
        "--test_size", type=int, default=None, help="测试条数(用于调试)"
    )
    parser.add_argument(
        "--restore",
        type=lambda x: x.lower() == "true",
        default=True,
        help="是否恢复已有结果（默认 True）",
    )

    args = parser.parse_args()

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    # 加载数据
    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dataset = load_jsonl(data_path)
    print(f"Loaded {len(dataset)} items from {data_path}")

    # 测试模式
    test_mode = args.test_size is not None
    if test_mode:
        dataset = dataset[: args.test_size]
        print(f"✓ Test mode: evaluating {len(dataset)} items")
        print(f"✓ Will output each item's model response\n")

    # 推理模型模式
    if args.reasoning_model:
        print(f"✓ Reasoning model mode enabled (thinking_budget=16384 auto-applied)")

    # 确定输出路径
    model_basename = os.path.basename(args.model)
    data_basename = data_path.stem  # 例如 "Chat"
    output_path = (
        Path("results") / f"run_baseline_{data_basename}_{model_basename}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nModel: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {output_path}\n")

    # Restore 模式：加载已有结果并过滤
    already_processed = []
    if args.restore:
        existing_results = load_existing_results(output_path)
        if existing_results:
            dataset, already_processed = filter_dataset_with_restore(
                dataset, existing_results
            )
            print(f"✓ Restore mode enabled:")
            print(f"  - Found {len(already_processed)} existing results")
            print(f"  - Need to process {len(dataset)} new items\n")
        else:
            print(f"✓ Restore mode enabled but no existing results found\n")
    else:
        print(f"✓ Restore mode disabled: processing all {len(dataset)} items\n")

    # 执行评估
    results = evaluate_dataset(
        client=client,
        model=args.model,
        dataset=dataset,
        max_tokens=args.max_tokens,
        is_reasoning_model=args.reasoning_model,
        test_mode=test_mode,
        num_threads=args.concurrency,
        output_path=output_path,
    )

    # 合并已有结果和新处理的结果
    if already_processed:
        final_results = already_processed + results
        print(
            f"\n✓ Merged {len(already_processed)} existing + {len(results)} new = {len(final_results)} total results"
        )
    else:
        final_results = results

    # 保存最终结果
    save_jsonl(final_results, output_path)
    print(f"\n✓ Results saved to: {output_path}")

    # 统计失败案例
    failed = sum(1 for r in final_results if r.get("score") is None)
    if failed > 0:
        print(f"\n⚠ Warning: {failed} items failed to parse")

    # 提示使用独立评估脚本
    print("\n" + "=" * 60)
    print("To analyze results, use the appropriate evaluation script:")
    print("  • RewardBench: python scripts/baseline/evaluate/evaluate_rewardbench.py")
    print("  • JudgeBench:  python scripts/baseline/evaluate/evaluate_judgebench.py")
    print("  • OffsetBias:  python scripts/baseline/evaluate/evaluate_offsetbias.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
