#!/bin/bash

# LLMBar-Adversarial 生成 key_focus_aspects 脚本

python generate_key_aspects.py \
    --model "qwq-32b" \
    --api_base "https://yunwu.ai/v1" \
    --api_key "sk-P2O0bcaVTVKHqKYsdIdPUslTCXYymDWHlGzRuWIcYi3s62Bc" \
    --concurrency 50 \
    # --test_size 10