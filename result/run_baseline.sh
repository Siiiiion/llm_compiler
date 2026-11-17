#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
nohup python -m LLM baseline --workload=bert_large \
 --input-shape='[1,128]' \
 --target=nvidia/geforce-rtx-4090 \
 --backend=graph \
 --num-trials=2000 \
 --output-log=bert_large_1x128_4090.json > bert_large_1x128_4090_tuning.log 2>&1 &
