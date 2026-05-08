nohup python /home/qsy/workspace/complier/llm_compiler/LLM/baseline.py --workload=bert_large \
 --input-shape='[1,128]' \
 --target=nvidia/geforce-rtx-4090 \
 --backend=graph \
 --num-trials=2000 \
 --output-log=bert_large_1x128_4090.json > bert_large_1x128_4090_tuning.log 2>&1 &