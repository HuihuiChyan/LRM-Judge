#!/bin/bash

# 可选参数: 添加 --test_size N 来指定测试数据条数（不指定则测试全部数据）
# 例如: --test_size 10

model="deepseek-ai/DeepSeek-R1"

python scripts/self_synthesized/run_self_synthesized.py \
    --model $model \
    --api_key "sk-scfdkliuvgltrqylrvkrqwjkbslazoeicflcocwdsgutuybo" \
    --api_base "https://api.siliconflow.cn/v1" \
    --data_file "datasets/offsetbias/offsetbias_`basename $model`.jsonl" \
    --concurrency 16 \
    --reasoning_model
    # --max_tokens 4096 \
    # --test_size 20
    # --reasoning_model
