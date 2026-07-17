# Installation

- Build and install this repo following the [guide](docs/install/from_source.rst).
- You can refer to my installation environment [here](version.log).
- [Note] To avoid this [issue](https://github.com/apache/tvm/issues/9362), please remember to set this👉. If you are a PyTorch user, it is recommended to set `(USE_LLVM "/path/to/llvm-config --link-static")` and `set(HIDE_PRIVATE_SYMBOLS ON)` to avoid potential symbol conflicts between different versions LLVM used by TVM and PyTorch.
- TLM uses huggingface for training and needs to install dependencies:
  ```shell
  pip install -r requirements.txt
  ```
- You can download the [tlm_dataset](https://drive.google.com/file/d/1MdOxSIBFqYl1pWWUmj18vm4AG4Sbly8Z/view) we have collected and put them in the corresponding path.
  ```
  (py38) ➜  ~ tree tlm_dataset -L 2
  tlm_dataset
  ├── gen
  │   ├── dataset
  │   ├── gen_data
  │   └── utils.json
  └── meta
      ├── dataset
      ├── meta_data
      └── meta_utils.json

  7 directories, 2 files
  ```

## Getting Started Instructions

To get started quickly, you need to download [tlm_dataset](https://drive.google.com/file/d/1omj7AfPFIjuSgs-UBld0YRKWmmCe08G3/view?usp=drive_link). Here we take compiling `bert_base` as an example.

```shell
cd gen
```

- Optional. We have already trained TLM in `gen_data/v100_gen_best`. You can repeat the following command, which will overwrite `gen_data/v100_gen_best`. When executing `run_train_clm_best_v100.py`, you need to 1) adjust `CUDA_VISIBLE_DEVICES` and `--per_device_train_batch_size` and 2) `apt install tmux`.
  ```shell
  python postprocess.py --target=nvidia/nvidia-v100

  python make_dataset.py \
  --for_type=for_gen_best_all \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/measure_records/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_best

  python run_train_clm_best_v100.py
  ```
- To generate tensor programs for bert_base, generate prompts first.
  ```shell
  python make_dataset.py \
  --for_type=for_gen_eval_sketch_only_bert \
  --target=nvidia/nvidia-v100 \
  --dataset_path=dataset/to_measure_programs/v100 \
  --tokenizer_path=gen_data/gen_tokenizer_v100 \
  --save_path=gen_data/v100_gen_eval_only_bert \
  --keep_cnt=64
  ```
- Generate tensor programs for bert_base.
  ```shell
  CUDA_VISIBLE_DEVICES=3 python gen_state.py \
  --model_name_or_path=gen_data/clm_gen_best_v100 \
  --sketch_path=gen_data/v100_gen_eval_only_bert/0_merge.json \
  --save_path=gen_data/v100_gen_eval_only_bert/gen_eval.json \
  --allow_repeat=True \
  --target=nvidia/nvidia-v100 \
  --keep_cnt=32
  ```
- Measure the execution latency of the generated tensor program. Ensure that the measurement hardware is exclusive to avoid inaccurate results.
  ```shell
  CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_eval_only_bert/gen_eval.json --measured-path=measured_only_bert.json
  ```
- Select the best-performing programs from the measured tensor programs to compile `bert_base` and measure the end-to-end execution latency of `bert_base`. Ensure that the measurement hardware is exclusive to avoid inaccurate results.
  ```shell
  CUDA_VISIBLE_DEVICES=3 TLM_LOG_FILE=measured_only_bert.json python tune_relay.py --workload=bert_base --input-shape=\[1,128\] --target=nvidia/nvidia-v100 --backend=graph
  ```

## TLM-Ansor

```shell
cd gen
```

1. Train TLM-base, taking the NVIDIA V100 as an example.
  - Partition the workload into subgraphs and save the results to the path `dataset/network_info/v100`.
     `--target`  can be found in `src/target/tag.cc` for other hardware. For CPUs, it can be set to something like this `--target="llvm -mcpu=core-avx2 -model=i7"`.
  - Dump tensor programs for those subgraphs. The resulting tensor programs have not measured execution latency and are unlabeled data. They will be saved to the path `dataset/to_measure_programs/v100`.
    ```shell
    python dump_programs.py --target=nvidia/nvidia-v100
    ```
  - Use unlabeled data to build a vocabulary and a tokenizer and save them to `--tokenizer_path`.
    ```shell
    python make_dataset.py \
    --for_type=for_gen_tokenizer \
    --target=nvidia/nvidia-v100 \
    --dataset_path=dataset/to_measure_programs/v100 \
    --tokenizer_path=gen_data/gen_tokenizer_v100
    ```
  - Use the tokenizer to train the TLM-base pre-train dataset and save it to `--save_path`.
    ```shell
    python make_dataset.py \
    --for_type=for_gen \
    --target=nvidia/nvidia-v100 \
    --dataset_path=dataset/to_measure_programs/v100 \
    --tokenizer_path=gen_data/gen_tokenizer_v100 \
    --save_path=gen_data/v100_gen_2154
    ```
  - Pre-train TLM-base. Adjust parameters such as `batch_size` in the run_train_clm.py file according to the GPU memory size. Requires `apt install tmux`.
    ```shell
    python run_train_clm.py
    ```
2. Train the TLM using iterative optimization. We provide two methods: the script with one kick and the step-by-step command.
  A. The script with one kick. The script uses a pipeline system and requires two machines. Both machines need to clone the TLM repository. The two machines communicate and exchange data using ssh and rsync.
  - On the training machine, 1) configure ssh password-free login, and then configure the target machine in `~/.ssh/config`, 2) set `device_id_all` in run.py to specify the GPU card IDs that can be used to train TLM; 3) Set `ssh_target` in run.py.
    ```shell
    python run.py --target=nvidia/nvidia-v100 --for_type=for_finetuning --finetuning_init=True
    ```
  - On the measurement machine, configure `available_ids` to specify the GPU card IDs that can be used for measurement.
    ```shell
    python run_measure.py
    ```
   B. Step-by-step command.
  - Before using TLM-base/TLM to generate tensor programs, generate prompts first.
    ```shell
    python make_dataset.py \
    --for_type=for_gen_train_sketch \
    --target=nvidia/geforce-rtx-4090 \
    --dataset_path=/data/qsy/workspace/dataset/to_measure_programs/4090 \
    --tokenizer_path=/data/qsy/huggingface/model/Qwen3-0.6B-fintuned \
    --save_path=/data/qsy/workspace/gen_data/qwen_4090_gen_train \
    --keep_cnt=48 \
    --test_file_idx=0
    ```
  - Qwen3-0.6B seq128 ONNX workload workflow.

    The Qwen3 experiment has its own workload registry. Do not use the generic
    `network_info/4090` registry for these records. Run the following commands from
    `/home/qsy/workspace/complier/llm_compiler/LLM`.

    First, generate the sketch prompts. `--test_file_idx=0` selects the current quarter
    of the workload files, which produces three workload groups in `0_merge.json`.

    ```shell
    cd /home/qsy/workspace/complier/llm_compiler/LLM

    /home/qsy/anaconda3/envs/tlm/bin/python make_dataset.py \
      --for_type=for_gen_train_sketch \
      --target=nvidia/geforce-rtx-4090 \
      --dataset_path=/home/qsy/workspace/dataset/to_measure_programs/qwen3_0_6b_seq128 \
      --tokenizer_path=/home/qsy/huggingface/model/Qwen3-0.6B-fintuned \
      --save_path=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train \
      --keep_cnt=48 \
      --test_file_idx=0 \
      --network_info_dir=/home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128 \
      --to_measure_program_dir=/home/qsy/workspace/dataset/to_measure_programs/qwen3_0_6b_seq128 \
      --skip_hold_out=True
    ```

    Generate build-valid states. The command deliberately writes to the existing
    `gen_train.json`. The implementation writes worker parts to a temporary directory and
    atomically replaces `gen_train.json` only after every worker succeeds. With
    `--require_full_keep_cnt=True`, a short workload leaves the old output untouched.
    Candidates are always deduplicated within this run; `--allow_repeat=True` only allows
    candidates that may have appeared in an older measurement round.

    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    /home/qsy/anaconda3/envs/tlm/bin/python gen_state.py \
      --target=nvidia/geforce-rtx-4090 \
      --model_name_or_path=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
      --sketch_path=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/0_merge.json \
      --save_path=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/gen_train.json \
      --allow_repeat=True \
      --keep_cnt=16 \
      --is_build=True \
      --max_retries=20 \
      --require_full_keep_cnt=True \
      --do_sample=True \
      --disable_eos_stop=False \
      --min_gen_tokens=256 \
      --gen_token_scale=4.0 \
      --generation_batch_size=32 \
      --sample_top_k=0 \
      --sample_top_p=1.0 \
      --sample_temperature=0.6 \
      --network_info_dir=/home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128
    ```

    Generation statistics are written next to the output as
    `gen_train.json.stats.json`. Inspect each workload before measurement:

    ```shell
    jq '.workloads[] | {
      workload_key,
      generation_rounds,
      parsed_states,
      build_attempted,
      build_valid,
      build_error_counts,
      duplicates,
      unique_accepted,
      kept,
      shortfall
    }' /home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/gen_train.json.stats.json

    wc -l /home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/gen_train.json
    ```

    For three workloads and `keep_cnt=16`, the expected output is 48 unique,
    build-valid records. If the strict run reports a shortfall, inspect the statistics and
    retry with `--max_retries=40`; do not disable build validation to fill the count.

    Measure the generated states on one exclusive RTX 4090. This overwrites the old
    `measured_gen_train.json`, which is required because it refers to the previous
    `gen_train.json` candidates.

    ```shell
    CUDA_VISIBLE_DEVICES=3 \
    /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
      --batch-size=64 \
      --target=nvidia/geforce-rtx-4090 \
      --network-info-dir=/home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128 \
      --to-measure-path=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/gen_train.json \
      --measured-path=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/measured_gen_train.json \
      --no-resume
    ```

    Validate the measurement file:

    ```shell
    jq -s '{
      total: length,
      valid: ([.[] | select(
        (.r[1] == 0) and any(.r[0][]; . > 0 and . < 1000000000)
      )] | length),
      errors: (group_by(.r[1]) | map({error_no: .[0].r[1], count: length}))
    }' /home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/measured_gen_train.json
    ```

    A cost of `1e10` is TVM's failure sentinel, not latency. Such a record has a non-zero
    `r[1]` error code and must not be used for tuning or training.

    For workloads with very large inputs, such as an FP16 vocabulary projection with a
    roughly 500 MiB weight tensor, the default five-second `LocalRunner` timeout can expire
    while the runner prepares inputs. Diagnose one record before treating `error_no=4` as an
    invalid CUDA schedule:

    ```shell
    CUDA_VISIBLE_DEVICES=3 \
    /home/qsy/anaconda3/envs/tlm/bin/python measure_programs.py \
      --batch-size=1 \
      --run-timeout=60 \
      --print-error-details \
      --target=nvidia/geforce-rtx-4090 \
      --network-info-dir=/path/to/network_info \
      --to-measure-path=/path/to/candidates.json \
      --measured-path=/path/to/measured_candidates.json \
      --no-resume
    ```

    A printed `TimeoutError` means the runner deadline was too short. Increase
    `--run-timeout`; do not fabricate or relabel the failed record. CUDA launch, allocation,
    and correctness errors require separate investigation even though they may share error
    number 4.

    To choose workloads for the next generation round, run the Qwen-aware task scheduler:

    ```shell
    /home/qsy/anaconda3/envs/tlm/bin/python task_scheduler.py \
      --target=nvidia/geforce-rtx-4090 \
      --network_info_dir=/home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128 \
      --to_measure_program_dir=/home/qsy/workspace/dataset/to_measure_programs/qwen3_0_6b_seq128 \
      --history_files=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/measured_gen_train.json \
      --output_path=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/task_scheduler_4090.pkl \
      --top_k=3 \
      --records_per_task=16
    ```

    Do not run the legacy `gen/postprocess.py` for this Qwen workflow. It reads file paths
    from `gen/utils.json` and deletes the existing JSON files under the generic
    `measure_records/4090` directory before rebuilding them.

  - Whole-model Relay/ONNX benchmark (model-independent).

    `LLM/benchmark_relay.py` measures the same fixed-shape graph in two modes:
    TVM Default (`opt_level=3`) and AutoScheduler (`ApplyHistoryBest`). It generates one
    shared input set, checks every optimized output against the baseline, excludes compile
    time from inference timing, and writes task coverage, compile time, timing samples,
    correctness, error percentiles, RMSE, cosine similarity, logits argmax agreement, and
    whole-model speedup to a JSON report. Optimized benchmarking is refused
    unless every task extracted from the exact Relay module has at least one valid history
    record. `error_no != 0`, zero cost, and costs at or above `1e9` seconds are invalid.

    For Qwen3-0.6B seq128, run this only after the measured history covers all 11 extracted
    tasks. The existing Qwen Relay artifact is an old-format artifact with an empty parameter
    dictionary and remains supported:

    ```shell
    cd /home/qsy/workspace/complier/llm_compiler/LLM

    CUDA_VISIBLE_DEVICES=3 \
    /home/qsy/anaconda3/envs/tlm/bin/python benchmark_relay.py \
      --relay-pickle='/home/qsy/workspace/dataset/network_info/qwen3_0_6b_seq128/(onnx_qwen3_0_6b_seq128,[(attention_mask,[1,128]),(input_ids,[1,128])]).relay.pkl' \
      --history=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/measured_gen_train.json \
      --target=nvidia/geforce-rtx-4090 \
      --baseline-lib=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/qwen3_default.so \
      --optimized-lib=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/qwen3_autoscheduler.so \
      --report=/home/qsy/workspace/gen_data/qwen3_0_6b_seq128_gen_train/whole_model_benchmark.json \
      --warmup=5 \
      --number=1 \
      --repeat=10 \
      --min-repeat-ms=500
    ```

    `--baseline-lib` and `--optimized-lib` are caches: a missing file is built and exported;
    an existing file is loaded without recompilation. Delete or choose new cache paths after
    changing the model, target, TVM build, or tuning history. The reported whole-model speedup
    is `baseline median latency / optimized median latency`.

    Any fixed-shape ONNX model can be benchmarked directly. Multiple input shapes and dtypes
    are JSON maps, so this is not tied to Qwen input names:

    ```shell
    CUDA_VISIBLE_DEVICES=0 \
    /home/qsy/anaconda3/envs/tlm/bin/python benchmark_relay.py \
      --onnx-model=/path/to/model.onnx \
      --input-shapes='{"tokens":[1,128],"mask":[1,128]}' \
      --input-dtypes='{"tokens":"int64","mask":"int64"}' \
      --history=/path/to/measured_history.json \
      --target=nvidia/geforce-rtx-4090 \
      --report=/path/to/whole_model_benchmark.json
    ```

    Use `--input-npz=/path/to/inputs.npz` when generated inputs are not semantically valid for
    a model. The NPZ must contain exactly one correctly shaped and typed array per model input.
    `--mode=baseline` or `--mode=optimized` can run one side independently. The escape hatch
    `--allow-incomplete-history` permits AutoScheduler fallback schedules, but its result is
    not a valid full-coverage tuning speedup and should not be used for the final comparison.

    New Relay artifacts written by `gen/dump_network_info.py` use a versioned format and retain
    the actual serialized parameter bytes. Older artifacts stored only the parameter byte
    count; they are usable only when that count denotes an empty parameter dictionary. For a
    non-empty legacy artifact, re-dump from the original model with `--overwrite-relay`.
  - Use TLM-base/TLM to generate tensor programs, `--model_name_or_path` specifies whether to use TLM-base or TLM.
    ```shell
    nohup bash -lc '
    cd /home/qsy/workspace/complier/llm_compiler/LLM && \
    CUDA_VISIBLE_DEVICES=1,2,3 /home/qsy/anaconda3/envs/tlm/bin/python gen_state.py \
      --target=nvidia/geforce-rtx-4090 \
      --model_name_or_path=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1 \
      --sketch_path=/home/qsy/workspace/gen_data/4090_gen_train/0_merge.json \
      --save_path=/home/qsy/workspace/gen_data/4090_gen_train/gen_train.json \
      --allow_repeat=True \
      --keep_cnt=16 \
      --is_build=False \
      --do_sample=True \
      --disable_eos_stop=False \
      --min_gen_tokens=256 \
      --gen_token_scale=4.0 \
      --generation_batch_size=32 \
      --sample_top_k=0 \
      --sample_top_p=1.0 \
      --sample_temperature=0.6
    ' > /home/qsy/workspace/complier/llm_compiler/LLM/gen_state_full.log 2>&1 &
    ```
  - Measure the execution latency of the generated tensor program and 'manually' update the path of the measurement results to the `utils.json` file. There are many errors in the initial measurement data of iterative optimization. These errors are normal and will gradually decrease as the iteration proceeds.
    ```shell
    CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_train/gen_train.json --measured-path=gen_data/measure_data_v100/finetuning_0.json
    ```
  - Organize the measured programs into `dataset/measure_records/v100`.
    ```shell
    python postprocess.py --target=nvidia/nvidia-v100
    ```
  - Build an SFT dataset.
    ```shell
    python make_dataset.py \
    --for_type=for_gen_best \
    --target=nvidia/nvidia-v100 \
    --dataset_path=dataset/measure_records/v100 \
    --tokenizer_path=gen_data/gen_tokenizer_v100 \
    --save_path=gen_data/v100_gen_best
    ```
  - SFT TLM-base.
    ```shell
    python run_train_clm_best_v100.py
    ```
3. Evaluation on target workload.
  - Generate prompts.
  - Generate tensor programs.
    ```shell
    CUDA_VISIBLE_DEVICES=4 python gen_state.py \
    --model_name_or_path=gen_data/clm_gen_best_v100 \
    --sketch_path=gen_data/v100_gen_eval/0_merge.json \
    --save_path=gen_data/v100_gen_eval/gen_eval.json \
    --allow_repeat=True \
    --target=nvidia/nvidia-v100 \
    --keep_cnt=32
    ```
  - Measure the execution latency of the generated tensor program.
    ```shell
    CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_eval/gen_eval.json --measured-path=gen_data/measure_data_v100/0_test_3.json
    ```
  - Use scripts to analyze the speedups.
    ```shell
    python speedup_eval.py --target=nvidia/nvidia-v100 --for_test=True
    ```
4. When the tuning budget is ample, we continue to optimize TLM using the target workload data. There are also two ways.
  A. The script with one kick.
  - On the training machine.
    ```shell
    python run.py --target=nvidia/nvidia-v100 --for_type=for_testtuning --testtuning_init=True
    ```
  - On the measurement machine.
    ```shell
    python run_measure.py
    ```
   B. Step-by-step command.
  - Generate prompts.
    ```shell
    python make_dataset.py \
    --for_type=for_gen_evaltuning_sketch \
    --target=nvidia/nvidia-v100 \
    --dataset_path=dataset/to_measure_programs/v100 \
    --tokenizer_path=gen_data/gen_tokenizer_v100 \
    --save_path=gen_data/v100_gen_evaltuning \
    --keep_cnt=64
    ```
  - Generate tensor programs.
    ```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3 python gen_state.py \
    --model_name_or_path=gen_data/clm_gen_best_v100 \
    --sketch_path=gen_data/v100_gen_evaltuning/0_merge.json \
    --save_path=gen_data/v100_gen_evaltuning/gen_eval.json \
    --allow_repeat=True \
    --target=nvidia/nvidia-v100 \
    --keep_cnt=32
    ```
  - Measure the execution latency of the generated tensor program and 'manually' update the path of the measurement results to the `utils.json` file.
    ```shell
    CUDA_VISIBLE_DEVICES=3 python measure_programs.py --batch-size=64 --target=nvidia/nvidia-v100 --to-measure-path=gen_data/v100_gen_evaltuning/gen_eval.json --measured-path=gen_data/measure_data_v100/testtuning_0.json
    ```
  - Organize the measured programs into `dataset/measure_records/v100`.
    ```shell
    python postprocess.py --target=nvidia/nvidia-v100
    ```
  - Build an SFT dataset.
    ```shell
    python make_dataset.py \
    --for_type=for_gen_best_all \
    --target=nvidia/nvidia-v100 \
    --dataset_path=dataset/measure_records/v100 \
    --tokenizer_path=gen_data/gen_tokenizer_v100 \
    --save_path=gen_data/v100_gen_best
    ```
  - SFT TLM-base
    ```shell
    python run_train_clm_best_v100.py
    ```
  - Not every task has the same optimization space. We use the task scheduler to allocate the tuning budget.
    ```shell
    python task_sheduler.py --target=nvidia/nvidia-v100 --for_testtuning=True
    ```

## TLM-Meta

```shell
cd meta
```

Similar to TLM-Ansor, command lines can be found in run.sh and run.py.
