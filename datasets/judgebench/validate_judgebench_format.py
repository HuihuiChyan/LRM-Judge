#!/usr/bin/env python3
"""
验证JudgeBench处理后的数据格式是否与run_self_synthesized.py兼容

用法:
    python validate_judgebench_format.py datasets/judgebench/judgebench_QwQ-32B.jsonl
"""

import json
import sys
from pathlib import Path


def validate_data_item(item: dict, index: int) -> list:
    """验证单条数据是否包含所有必需字段"""
    errors = []
    required_fields = ["prompt", "chosen", "rejected", "key_focus_aspects"]

    for field in required_fields:
        if field not in item:
            errors.append(f"Item {index}: Missing required field '{field}'")
        elif field == "key_focus_aspects":
            if not isinstance(item[field], list):
                errors.append(f"Item {index}: 'key_focus_aspects' must be a list")
            elif not item[field]:
                errors.append(f"Item {index}: 'key_focus_aspects' is empty")
        elif not item[field]:
            errors.append(f"Item {index}: Field '{field}' is empty")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_judgebench_format.py <jsonl_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    print(f"Validating {file_path}...")
    print("-" * 60)

    all_errors = []
    total_items = 0
    valid_items = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            total_items += 1
            try:
                item = json.loads(line)
                errors = validate_data_item(item, i)

                if errors:
                    all_errors.extend(errors)
                else:
                    valid_items += 1

                # 展示第一条数据的结构
                if i == 1:
                    print("Sample data structure:")
                    print(f"  Fields: {list(item.keys())}")
                    print(f"  prompt: {item.get('prompt', '')[:80]}...")
                    print(f"  chosen: {item.get('chosen', '')[:80]}...")
                    print(f"  rejected: {item.get('rejected', '')[:80]}...")
                    print(f"  key_focus_aspects: {item.get('key_focus_aspects', [])}")
                    print(f"  source: {item.get('source', 'N/A')}")
                    print("-" * 60)

            except json.JSONDecodeError as e:
                all_errors.append(f"Line {i}: JSON decode error - {e}")

    # 输出验证结果
    print(f"\nValidation Results:")
    print(f"  Total items: {total_items}")
    print(f"  Valid items: {valid_items}")
    print(f"  Invalid items: {total_items - valid_items}")

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} error(s):")
        for error in all_errors[:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
        sys.exit(1)
    else:
        print("\n✅ All data items are valid and compatible with run_self_synthesized.py!")
        print(f"\n✅ Data format check passed! Ready to use with run_self_synthesized.py")


if __name__ == "__main__":
    main()
