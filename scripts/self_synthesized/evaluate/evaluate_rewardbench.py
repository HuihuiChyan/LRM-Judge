#!/usr/bin/env python3
"""
RewardBench数据集评估脚本
从results目录读取run_self_synthesized.py的输出,按section计算准确率

用法:
    python evaluate_rewardbench.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


# 配置部分 - 直接在此修改文件路径
RESULT_FILE = "results/run_synthesized_rewardbench_DeepSeek-R1_DeepSeek-R1.jsonl"
SECTIONS = ["Chat", "Chat Hard", "Safety", "Reasoning"]


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calculate_section_stats(results: List[Dict]) -> Dict[str, Dict]:
    """
    按section计算统计数据

    Returns:
        {
            "Chat": {"accuracy": 0.85, "total": 100, "correct": 85},
            "Chat Hard": {...},
            ...
        }
    """
    section_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        section = result.get("section", "Unknown")
        score = result.get("score")

        if score is not None:
            section_stats[section]["total"] += 1
            if score == 1:  # chosen被判断为更好
                section_stats[section]["correct"] += 1

    # 计算准确率
    final_stats = {}
    for section, stats in section_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        final_stats[section] = {
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
    section_stats = calculate_section_stats(results)

    # 输出结果
    print("=" * 60)
    print("REWARDBENCH EVALUATION RESULTS")
    print("=" * 60)

    total_correct = 0.0
    total_count = 0

    for section in SECTIONS:
        if section in section_stats:
            stats = section_stats[section]
            print(f"\n{section}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Correct: {stats['correct']} / {stats['total']}")
            total_correct += stats["correct"]
            total_count += stats["total"]
        else:
            print(f"\n{section}: No data")

    # 总体准确率
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    print(f"\n{'─' * 60}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Total Correct: {total_correct} / {total_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
