#!/usr/bin/env python3
"""
JudgeBench数据集评估脚本
从results目录读取run_self_synthesized.py的输出,按source类别计算准确率

Source分类规则:
- mmlu-pro-*: 合并为 mmlu-pro
- livebench-reasoning: 保持不变
- livebench-math: 保持不变
- livecodebench: 保持不变

用法:
    python evaluate_judgebench.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


# 配置部分 - 直接在此修改文件路径
RESULT_FILE = "results/run_synthesized_judgebench_DeepSeek-R1_DeepSeek-R1.jsonl"

# 定义子集分类规则
SUBSET_CATEGORIES = {
    "mmlu-pro": "mmlu-pro-",  # 所有以mmlu-pro-开头的source归为mmlu-pro
    "livebench-reasoning": "livebench-reasoning",
    "livebench-math": "livebench-math",
    "livecodebench": "livecodebench",
}

# 输出顺序
SUBSET_ORDER = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench"]


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def categorize_source(source: str) -> str:
    """
    将原始source映射到子集类别

    Args:
        source: 原始source字段,如 "mmlu-pro-law", "livebench-reasoning"

    Returns:
        子集类别名称,如 "mmlu-pro", "livebench-reasoning"
    """
    # 检查是否以mmlu-pro-开头
    if source.startswith("mmlu-pro-"):
        return "mmlu-pro"

    # 其他类别直接匹配
    for category, pattern in SUBSET_CATEGORIES.items():
        if category != "mmlu-pro" and source == pattern:
            return category

    return "other"  # 未知类别


def calculate_subset_stats(results: List[Dict]) -> Dict[str, Dict]:
    """
    按子集类别计算统计数据

    Returns:
        {
            "mmlu-pro": {"accuracy": 0.85, "total": 100, "correct": 85.0},
            "livebench-reasoning": {...},
            ...
        }
    """
    subset_stats = defaultdict(lambda: {"correct": 0.0, "total": 0})

    for result in results:
        source = result.get("source", "unknown")
        subset = categorize_source(source)
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
    print("JUDGEBENCH EVALUATION RESULTS")
    print("=" * 60)

    total_correct = 0.0
    total_count = 0

    for subset in SUBSET_ORDER:
        if subset in subset_stats:
            stats = subset_stats[subset]
            print(f"\n{subset}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Correct: {stats['correct']:.1f} / {stats['total']}")
            total_correct += stats["correct"]
            total_count += stats["total"]
        else:
            print(f"\n{subset}: No data")

    # 显示其他未分类的数据(如果有)
    if "other" in subset_stats:
        stats = subset_stats["other"]
        print(f"\nother (unclassified):")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        print(f"  Correct: {stats['correct']:.1f} / {stats['total']}")
        total_correct += stats["correct"]
        total_count += stats["total"]

    # 总体准确率
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n{'─' * 60}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Total Correct: {total_correct:.1f} / {total_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
