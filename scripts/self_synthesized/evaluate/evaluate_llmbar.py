#!/usr/bin/env python3
"""
LLMBar-Adversarial数据集评估脚本
从results目录读取run_self_synthesized.py的输出,按subset类型计算准确率

Subset类型:
- neighbor: 相近指令对抗 (使用相近但不同的指令生成对抗输出)
- gptinst: GPT-4指令变体 (GPT-4生成的相似指令变体)
- gptout: GPT-4表面优质输出 (GPT-4生成的表面优质但无用/错误的输出)
- manual: 人工对抗样本 (人工构造的对抗样本)

用法:
    python evaluate_llmbar.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


# 配置部分 - 直接在此修改文件路径
RESULT_FILE = "results/run_synthesized_llmbar_qwq-32b_qwq-32b.jsonl"

# 定义subset顺序
SUBSET_ORDER = ["neighbor", "gptinst", "gptout", "manual"]


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calculate_subset_stats(results: List[Dict]) -> Dict[str, Dict]:
    """
    按subset类型计算统计数据

    Returns:
        {
            "neighbor": {"accuracy": 0.85, "total": 134, "correct": 114.0},
            "gptinst": {...},
            ...
        }
    """
    subset_stats = defaultdict(lambda: {"correct": 0.0, "total": 0})

    for result in results:
        subset = result.get("subset", "unknown")
        score = result.get("score")

        if score is not None:
            subset_stats[subset]["total"] += 1
            if score == 1:  # chosen被判断为更好
                subset_stats[subset]["correct"] += 1
            elif score == 0.5:  # 平局计为0.5分
                subset_stats[subset]["correct"] += 0.5

    # 计算准确率
    final_stats = {}
    for subset, stats in subset_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        final_stats[subset] = {
            "accuracy": accuracy,
            "total": stats["total"],
            "correct": stats["correct"],
        }

    return final_stats


def main():
    # 加载结果文件
    result_path = Path(RESULT_FILE)
    if not result_path.exists():
        print(f"❌ Error: Result file not found: {result_path}")
        print(f"Please update RESULT_FILE in the script to point to your result file.")
        return

    results = load_jsonl(result_path)
    print(f"Loaded {len(results)} results from {result_path.name}\n")

    # 计算统计
    subset_stats = calculate_subset_stats(results)

    # 输出结果
    print("=" * 60)
    print("LLMBAR-ADVERSARIAL EVALUATION RESULTS")
    print("=" * 60)

    total_correct = 0.0
    total_count = 0

    for subset in SUBSET_ORDER:
        if subset in subset_stats:
            stats = subset_stats[subset]
            print(f"\n{subset.upper()}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Correct: {stats['correct']:.1f} / {stats['total']}")
            total_correct += stats["correct"]
            total_count += stats["total"]
        else:
            print(f"\n{subset.upper()}: No data")

    # 显示其他未分类的subset(如果有)
    for subset in subset_stats:
        if subset not in SUBSET_ORDER:
            stats = subset_stats[subset]
            print(f"\n{subset.upper()} (unclassified):")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Correct: {stats['correct']:.1f} / {stats['total']}")
            total_correct += stats["correct"]
            total_count += stats["total"]

    # 总体准确率
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n{'─' * 60}")
    print(f"OVERALL ACCURACY: {overall_accuracy:.2%}")
    print(f"Total Correct: {total_correct:.1f} / {total_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
