#!/usr/bin/env python3
"""
Self-Synthesized Judge Model Evaluation Script
基于动态生成的评估重点关键词测试 LLM-as-a-judge 模型的性能

用法:
    # 普通模型
    python run_self_synthesized.py --model "gpt-4o" --api_key "xxx" --api_base "https://api.openai.com/v1" --data_file "datasets/section_jsonl/DeepSeek-V3.jsonl" --concurrency 10

    # 推理模型
    python run_self_synthesized.py --model "deepseek-reasoner" --api_key "xxx" --api_base "https://api.deepseek.com" --data_file "datasets/section_jsonl/DeepSeek-V3.jsonl" --reasoning_model --concurrency 5
"""

import argparse
import asyncio
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

# API 重试配置（参考 rewardbench/generative.py）
API_MAX_RETRY = 10
API_RETRY_SLEEP = 20
API_ERROR_OUTPUT = "$ERROR$"

# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are an impartial judge evaluating AI assistant responses."""

USER_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should be performed by following the provided evaluation plan step-by-step. Avoid copying the plan when doing the evaluation. Please also only stick to the given plan and provide explanation of how the plan is executed to compare the two responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your evaluation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{prompt}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

[The Start of Evaluation Plan]
{evaluation_plan}
[The End of Evaluation Plan]"""


# ============================================================================
# Core Functions
# ============================================================================


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict], file_path: Path):
    """保存JSONL文件（增量模式）"""
    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_existing_results(file_path: Path) -> Dict[str, Dict]:
    """
    加载已有结果文件，构建查找字典

    Returns:
        Dict[key, result]: key 由 prompt, chosen, rejected, evaluation_plan 四个字段生成
    """
    if not file_path.exists():
        return {}

    results_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # 生成唯一键
                key = _generate_item_key(item)
                results_dict[key] = item
    except Exception as e:
        print(f"Warning: Failed to load existing results from {file_path}: {e}")
        return {}

    return results_dict


def _generate_item_key(item: Dict) -> str:
    """
    根据 prompt, chosen, rejected, evaluation_plan 生成唯一键
    """
    import hashlib

    key_fields = [
        item.get("prompt", ""),
        item.get("chosen", ""),
        item.get("rejected", ""),
        item.get("evaluation_plan", ""),
    ]
    # 使用 JSON 序列化确保一致性，然后计算哈希
    key_str = json.dumps(key_fields, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def filter_dataset_with_restore(
    dataset: List[Dict], existing_results: Dict[str, Dict]
) -> Tuple[List[Dict], List[Dict]]:
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


def process_judgement(output: str) -> Tuple[Optional[int], str]:
    """
    从模型输出中提取评判结果

    Returns:
        (score, raw_output)
        score: "A" 表示 A 更好, "B" 表示 B 更好, None 表示解析失败
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


def extract_reasoning(output: str) -> Tuple[str, str]:
    """
    从输出中提取推理内容和实际响应

    Returns:
        (reasoning, content)
    """
    # 查找 <REASONING>...</REASONING> 标记
    reasoning_pattern = r"<REASONING>\s*(.*?)\s*</REASONING>"
    match = re.search(reasoning_pattern, output, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        content = re.sub(reasoning_pattern, "", output, flags=re.DOTALL).strip()
        return reasoning, content

    return "", output


def chat_completion_openai(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 16384,
    temperature: float = 0.0,
    is_reasoning_model: bool = False,
) -> Tuple[str, str]:
    """
    调用 OpenAI API 进行对话补全，支持自动重试

    参考 rewardbench/generative.py 的实现

    Args:
        is_reasoning_model: 如果为 True，自动添加 thinking_budget=16384

    Returns:
        (output, reasoning) - 如果是推理模型会分别返回输出和推理内容
    """
    output = API_ERROR_OUTPUT

    for retry_count in range(API_MAX_RETRY):
        try:
            # 准备请求参数
            request_params = {
                "model": model,
                "messages": messages,
                "n": 1,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # 如果是推理模型，自动添加 thinking_budget（参考 generative.py:896-913）
            if is_reasoning_model:
                request_params["extra_body"] = {"thinking_budget": 16384}

            response = client.chat.completions.create(**request_params)

            # 提取推理内容（参考 generative.py:919-926）
            msg = response.choices[0].message
            content = getattr(msg, "content", None) or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""

            # 如果有推理内容，将其包装到输出中
            if reasoning:
                output = f"<REASONING>\n{reasoning}\n</REASONING>\n{content}"
            else:
                output = content

            return output, reasoning

        except openai.APIError as e:
            print(f"[Retry {retry_count + 1}/{API_MAX_RETRY}] OpenAI API Error: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.APIConnectionError as e:
            print(f"[Retry {retry_count + 1}/{API_MAX_RETRY}] Connection Error: {e}")
            time.sleep(API_RETRY_SLEEP)

        except openai.RateLimitError as e:
            print(f"[Retry {retry_count + 1}/{API_MAX_RETRY}] Rate Limit Error: {e}")
            time.sleep(API_RETRY_SLEEP)

        except Exception as e:
            print(f"[Retry {retry_count + 1}/{API_MAX_RETRY}] Unexpected Error: {e}")
            time.sleep(API_RETRY_SLEEP)

    # 所有重试都失败
    print(f"❌ API call failed after {API_MAX_RETRY} retries")
    return API_ERROR_OUTPUT, ""


def get_judgement(
    client: OpenAI,
    model: str,
    item: Dict,
    max_tokens: int = 16384,
    is_reasoning_model: bool = False,
    test_mode: bool = False,
) -> Dict:
    """
    对单个数据项进行评判，支持解析失败时的自动重试

    Args:
        client: OpenAI 客户端
        model: 模型名称
        item: 数据项,包含 prompt, chosen, rejected, evaluation_plan 等字段
        max_tokens: 最大生成 token 数
        is_reasoning_model: 是否为推理模型(会自动添加 thinking_budget=16384)
        test_mode: 测试模式,如果为 True 会输出每条数据的模型输出

    Returns:
        包含评判结果的字典
    """
    # 随机打乱答案顺序以减少位置偏差（只在第一次生成时随机，重试时保持不变）
    shuffle = random.choice([True, False])
    if shuffle:
        response_a = item["rejected"]
        response_b = item["chosen"]
    else:
        response_a = item["chosen"]
        response_b = item["rejected"]

    # 获取评估计划
    evaluation_plan = item.get("evaluation_plan", "")
    if not evaluation_plan or evaluation_plan in ["parsing_failed", "api_error"]:
        evaluation_plan = "Evaluate the responses based on overall quality, accuracy, helpfulness, and appropriateness."

    # 构建提示
    user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt=item["prompt"],
        response_a=response_a,
        response_b=response_b,
        evaluation_plan=evaluation_plan,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 重试逻辑：当无法解析出有效判决时，最多重试 API_MAX_RETRY 次
    for retry_count in range(API_MAX_RETRY):
        # 调用模型
        output, reasoning = chat_completion_openai(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            is_reasoning_model=is_reasoning_model,
        )

        # 如果 API 调用失败，直接返回错误结果
        if output == API_ERROR_OUTPUT:
            return {
                **item,
                "model_output": output,
                "score": None,
                "shuffle": shuffle,
                "error": "API call failed after all retries",
            }

        # 测试模式下输出模型响应
        if test_mode:
            print("\n" + "=" * 80)
            print(
                f"📝 Item ID: {item.get('id', 'N/A')} | Section: {item.get('section', 'N/A')}"
            )
            if retry_count > 0:
                print(f"🔄 Retry attempt: {retry_count}/{API_MAX_RETRY - 1}")
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

        # 如果成功解析出 score，返回结果
        if score is not None:
            result = {
                **item,
                "model_output": raw_output,
                "score": score,
                "shuffle": shuffle,
            }
            # 如果有推理内容，单独提取
            if reasoning:
                result["reasoning"] = reasoning

            # 如果经过重试才成功，记录重试次数
            if retry_count > 0:
                result["retry_count"] = retry_count

            return result

        # 解析失败，打印警告并准备重试
        if retry_count < API_MAX_RETRY - 1:
            print(
                f"⚠️ [Retry {retry_count + 1}/{API_MAX_RETRY}] Failed to parse judgement, retrying..."
            )
            print(f"   Output snippet: {raw_output[:200]}...")
            time.sleep(API_RETRY_SLEEP)
        else:
            # 所有重试都失败，返回解析失败的结果
            print(
                f"❌ Failed to parse judgement after {API_MAX_RETRY} attempts, marking as failed"
            )

    # 所有重试都失败，返回解析失败的结果
    result = {
        **item,
        "model_output": raw_output if "raw_output" in locals() else output,
        "score": None,
        "shuffle": shuffle,
        "error": f"Failed to parse judgement after {API_MAX_RETRY} retries",
    }

    if "reasoning" in locals() and reasoning:
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
        test_mode: 测试模式，会输出每条数据的模型输出
        num_threads: 并发线程数
        output_path: 输出文件路径（用于增量保存）

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
        description="Self-Synthesized Judge Model Evaluation"
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
        help="是否为推理模型（自动启用 thinking_budget=16384）",
    )

    # 数据配置
    parser.add_argument(
        "--data_file", type=str, required=True, help="输入数据文件路径 (jsonl)"
    )

    # 执行配置
    parser.add_argument("--concurrency", type=int, default=10, help="并发线程数")
    parser.add_argument(
        "--test_size", type=int, default=None, help="测试条数（用于调试）"
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
    data_basename = data_path.stem  # 例如 "DeepSeek-V3"
    output_path = (
        Path("results") / f"run_synthesized_{data_basename}_{model_basename}.jsonl"
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
    print(
        "  • RewardBench: python scripts/self_synthesized/evaluate/evaluate_rewardbench.py"
    )
    print(
        "  • JudgeBench:  python scripts/self_synthesized/evaluate/evaluate_judgebench.py"
    )
    print(
        "  • OffsetBias:  python scripts/self_synthesized/evaluate/evaluate_offsetbias.py"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
