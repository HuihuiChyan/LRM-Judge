#!/bin/bash

# 可选参数:
# --test_size N: 指定测试数据条数（不指定则测试全部数据）
# --restore true/false: 是否恢复已有结果（默认 true）
# --reasoning_model: 是否为推理模型

model="qwq-32b"
api_key="put-your-key-here"

python scripts/self_synthesized/run_self_synthesized_plan.py \
    --model $model \
    --api_base "https://yunwu.ai/v1" \
    --api_key $api_key \
    --data_file "datasets/llmbar/llmbar_`basename $model`.jsonl" \
    --concurrency 40 \
    --restore true \
    --reasoning_model
    # --max_tokens 4096 \
    # --test_size 20
    # --reasoning_model
    # --api_base "https://api.siliconflow.cn/v1" \
