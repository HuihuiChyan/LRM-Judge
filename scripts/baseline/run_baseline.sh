#!/bin/bash

# 可选参数:
# --test_size N: 指定测试数据条数（不指定则测试全部数据）
# --restore true/false: 是否恢复已有结果（默认 true）
# --reasoning_model: 是否为推理模型

model="Qwen/Qwen2.5-32B-Instruct"

python scripts/baseline/run_baseline.py \
    --model $model \
    --api_key "sk-scfdkliuvgltrqylrvkrqwjkbslazoeicflcocwdsgutuybo" \
    --api_base "https://api.siliconflow.cn/v1" \
    --data_file "datasets/judgebench/judgebench_`basename $model`.jsonl" \
    --concurrency 20 \
    --restore true \
    --max_tokens 4096 \
    # --test_size 20
    # --reasoning_model
