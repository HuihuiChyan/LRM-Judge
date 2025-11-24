#!/bin/bash
# JudgeBench数据关键评估点生成脚本
# 用法: bash run_generate_key_aspects.sh

python generate_key_aspects.py \
    --api_base "https://api.siliconflow.cn/v1" \
    --api_key "sk-scfdkliuvgltrqylrvkrqwjkbslazoeicflcocwdsgutuybo" \
    --model "Qwen/QwQ-32B" \
    --concurrency 16 \
    --input_dir "." \
    --output_dir "." \
    # --test_size 10  # 取消注释以启用测试模式
