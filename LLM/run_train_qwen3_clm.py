import os
import subprocess


session_name = os.path.basename(os.path.abspath(__file__))
log_file = f"{session_name}.log"
session_name = session_name.replace(".", "_")
torchrun_bin = "/data/qsy/anaconda3/envs/tlm/bin/torchrun"
master_port = 29531

if os.path.exists(log_file):
    tag = input(log_file + " exist, delete it? [n]")
    if tag == "y":
        os.remove(log_file)

cmd = """tmux new -s %s -d '{
{
set -x
echo "#################################################################"
date

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,1,2,3 %s --nproc_per_node=4 --master_port=%s train_qwen3_clm.py \
                                    --do_train \
                                    --model_name_or_path=/data/qsy/huggingface/model/Qwen3-0.6B \
                                    --tokenizer_name=/data/qsy/huggingface/model/Qwen3-0.6B \
                                    --output_dir=/data/qsy/huggingface/model/Qwen3-0.6B-fintuned \
                                    --dataset_name=/data/qsy/workspace/gen_data/4090_gen_qwen \
                                    --per_device_train_batch_size=5 \
                                    --logging_steps=100 \
                                    --num_train_epochs=3 \
                                    --remove_unused_columns=False \
                                    --learning_rate=5e-5 \
                                    --save_steps=4000 \
                                    --save_total_limit=3 \
                                    --dataloader_num_workers=4 \
                                    --bf16=True \
                                    --gradient_checkpointing=True \
                                    --ddp_find_unused_parameters=False \
                                    --report_to=none

date
} |& tee -a %s
}'
""" % (session_name, torchrun_bin, master_port, log_file)

subprocess.Popen(cmd, shell=True)