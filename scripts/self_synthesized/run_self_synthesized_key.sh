#!/bin/bash

# 可选参数: 添加 --test_size N 来指定测试数据条数（不指定则测试全部数据）
# 例如: --test_size 10

model="qwq-32b"
api_key="put-your-key-here"

python scripts/self_synthesized/run_self_synthesized_key.py \
    --model $model \
    --api_base "https://yunwu.ai/v1" \
    --api_key $api_key \
    --data_file "datasets/llmbar/llmbar_key_`basename $model`.jsonl" \
    --concurrency 50 \
    --reasoning_model
    # --max_tokens 4096 \
    # --test_size 20
    # --reasoning_model
