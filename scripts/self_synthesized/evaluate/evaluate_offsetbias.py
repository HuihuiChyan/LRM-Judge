#!/usr/bin/env python3
"""
OffsetBias数据集评估脚本
从results目录读取run_self_synthesized.py的输出,按bias类型计算准确率

Bias类型:
- length bias: 长度偏差
- concreteness: 具体性偏差
- empty reference: 空引用
- content_continuation: 内容延续
- nested_instruction: 嵌套指令
- familiar knowledge preference bias: 熟悉知识偏好偏差

用法:
    python evaluate_offsetbias.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


# 配置部分 - 直接在此修改文件路径
RESULT_FILE = (
    "results/run_synthesized_offsetbias_deepseek-r1-250528_deepseek-r1-250528.jsonl"
)

# 定义bias类型顺序(按数据量排序)
BIAS_ORDER = [
    "length bias",
    "concreteness",
    "empty reference",
    "content_continuation",
    "nested_instruction",
    "familiar knowledge preference bias"
]


def load_jsonl(file_path: Path) -> List[Dict]:
    """加载JSONL文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def calculate_bias_stats(results: List[Dict]) -> Dict[str, Dict]:
    """
    按bias类型计算统计数据

    Returns:
        {
            "length bias": {"accuracy": 0.85, "total": 17, "correct": 14.5},
            "concreteness": {...},
            ...
        }
    """
    bias_stats = defaultdict(lambda: {"correct": 0.0, "total": 0})

    for result in results:
        bias = result.get("bias", "unknown")
        score = result.get("score")

        if score is not None:
            bias_stats[bias]["total"] += 1
            if score == 1:  # chosen被判断为更好
                bias_stats[bias]["correct"] += 1
            elif score == 0.5:  # 平局计为0.5分
                bias_stats[bias]["correct"] += 0.5

    # 计算准确率
    final_stats = {}
    for bias, stats in bias_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        final_stats[bias] = {
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
    bias_stats = calculate_bias_stats(results)

    # 输出结果
    print("=" * 60)
    print("OFFSETBIAS EVALUATION RESULTS")
    print("=" * 60)

    total_correct = 0.0
    total_count = 0

    for bias in BIAS_ORDER:
        if bias in bias_stats:
            stats = bias_stats[bias]
            print(f"\n{bias}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Correct: {stats['correct']:.1f} / {stats['total']}")
            total_correct += stats["correct"]
            total_count += stats["total"]
        else:
            print(f"\n{bias}: No data")

    # 显示其他未分类的bias(如果有)
    for bias in bias_stats:
        if bias not in BIAS_ORDER:
            stats = bias_stats[bias]
            print(f"\n{bias} (unclassified):")
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
