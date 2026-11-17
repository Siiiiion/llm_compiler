# 全量 CLM 训练数据重建方案

制定日期：2026-10-21。

2026-10-21 补充前置条件：原始 dump 并非全部经过构建验证，已经找到同一原始 state 后续编译失败的实例，见 [原始数据有效性核查](05_dump_dataset_validity_analysis.md)。本方案现有命令是“原样保留 raw 的完整重编码”，只能作为序列修复对照，不能据此宣称得到全部可构建的正式训练集。

若建立正式可构建训练集，应先验证原始 TVM 记录、过滤并保存验证清单，再生成新的 raw 与 Arrow；保留原有 workload 划分和 hold-out 隔离，过滤后样本数以通过清单为准。该门禁及支持断点续跑的审计入口已实现，入口和运行边界见 [第 5 份计划第 9 节](05_dump_dataset_validity_analysis.md#9-按本计划实施源记录门禁与后台审计)，不能用下文原样重编码命令替代源记录验证。正式重编码还需传入 `--build-audit-manifest` 绑定过滤后的 raw 摘要。

原样重编码对照可从现有完整 raw JSONL 重新编码，保留原有 workload 划分和全部训练样本，使用固定的扩词表 tokenizer、2048 长度上限和严格超长报错。重建必须以独立后台会话运行，退出终端、断开 SSH 或关闭查看日志的窗口后继续执行。对照数据通过下文验收后可用于受控实验；正式训练还需通过上述源记录门禁。

本文保留原样重编码对照的执行方案。正式数据的全量源记录构建审计，以及 2026-11-02 的 raw/Arrow 串联后台重建均已完成并通过数据验收；实际命令、产物路径和验收结果见 [第 5 份计划第 9.9 节](05_dump_dataset_validity_analysis.md#99-正式-rawarrow-后台重建2026-11-02)。第 3 步生成重建链路验证已提交独立后台任务，执行记录见该文第 9.10 节；GPU 训练尚未启动。

## 1. 本次重建的范围

本次修复训练数据的完整序列与 token 边界，产出供 `train/train_qwen3_clm.py` 使用的 Hugging Face DatasetDict。

- 从 `train_raw.json`、`validation_raw.json` 读取完整原始文本。文件虽然以 `.json` 结尾，实际按 JSONL 保存。
- 保留原始 train/validation 划分、行顺序与样本数量，以便与历史模型进行对照。
- 使用当前 `ppt_boundary_v2` 协议和固定扩词表 tokenizer。
- 每条序列完整编码，不截断；没有超过上限时不丢弃样本。
- CLM 继续保留原始文本的 EOS 语义；本轮不额外追加 EOS、不改变监督任务、不重新训练 tokenizer。

原始数据没有有效实测延迟，本轮重建得到的是结构学习数据。优秀调度 SFT、实测偏好数据扩充，以及 DPO/LAPO 的 JSONL 数据属于后续独立步骤。

本文“保持全部样本”的数量约束仅适用于原样重编码对照。正式过滤版必须另存目录与来源清单，逐 workload 报告通过率及覆盖损失，不应为凑齐原数量保留失败调度。

## 2. 已核实的输入与数据边界

输入目录：`/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext`。

| 项目 | train | validation |
| --- | ---: | ---: |
| 原始文件 | `train_raw.json` | `validation_raw.json` |
| 实际逐行核查样本数 | 2,169,636 | 111,611 |
| 规范化后 workload 数 | 2,061 | 108 |
| source_file 数 | 2,061 | 108 |
| 文件大小（字节） | 4,077,832,963 | 203,199,734 |
| 缺少独立 `PPT` 字段的文本数 | 0 | 0 |
| 标记为有实测延迟的样本数 | 0 | 0 |

合计 2,281,247 条，约 3.99 GiB 原始文件。本轮按规范化 workload key 检查：

- train 与 validation 的 workload 交集为 0。
- 使用 `LLM/core/common.py`、`cuda -model=4090` 对当前 registry 枚举出的 289 个 hold-out workload，两份 raw 与其交集均为 0。

这些结论适用于上述输入快照；更换输入、registry 或评测集合时需要重新核查。重建时不要把 `0_merge.json` 再随机切分，也不要把 eval-only 目录合入训练。

输入 SHA-256：

```text
train_raw.json
95f25088da19914870af056eb768dc887e7d85ee30f7ee94ad8f4cc9493a6d06

validation_raw.json
2a18ec41a7f9eea3add590bf7b20130e4fa2b355ab7c28e46db7b9fda2d91ec7
```

固定 tokenizer：

```text
/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2
```

本轮核实 `len(tokenizer)=151709`、模型 `vocab_size=151712`、`PPT id=151669`。数据、继续训练和生成都应显式指定这份 tokenizer。不要因模型名相近而自动换成原始 Qwen3 tokenizer。

## 3. 长度决策

本轮为每个 workload 选择首条、字符最短和字符最长记录，去除重复行号后检查了 6,159 条训练样本和 322 条验证样本。

| 代表样本编码结果 | train | validation |
| --- | ---: | ---: |
| 完整 token 长度最大值 | 1,193 | 1,179 |
| 完整 token 长度 p95 | 1,150 | 1,145 |
| 超过 1024 的代表样本数 | 576 | 27 |
| 超过 2048 的代表样本数 | 0 | 0 |
| 最短监督 suffix token 数 | 45 | 46 |

上述分布来自按 workload 选取的代表样本，不能当成全量样本的 p95 或全量最长值；字符最长记录也不保证 token 最长。它支持第一轮采用 2048，最终是否覆盖全部样本由全量编码的 `tokenization_stats.json` 决定。

建议参数：

```text
max_length = 2048
overlength_policy = error
ppt_marker = PPT
```

2048 是保存序列的上限；编码器使用动态长度，不会将所有样本补齐为 2048。训练时仍由 collator 按当前 batch 动态 padding。

如果全量扫描发现超过 2048：

1. 保留失败目录中的 `tokenization_stats.json`，检查超长样本数及受影响 workload。
2. 根据实际最大长度选择更大上限，并对长样本做训练显存验证；使用新的 run 目录重建。
3. 第一轮完整性对照不使用 `drop` 掩盖数据缺失。若后续决定建立丢弃版或分长度桶版本，单独命名并报告覆盖损失。

当前代码尚未实现按 TVM 中间状态自动拆分超长调度，本轮不能通过按 token 硬切块替代完整样本。

## 4. 执行入口与资源安排

使用 [rebuild_clm_dataset.py](../../data/rebuild_clm_dataset.py)，它复用当前修复后的编码和完整性检查，且不需要启动 TVM registry 或 GPU。

本轮无需重新运行 `make_dataset.py --for_type for_gen`。后者会重新遍历源调度、划分文件并清理目标目录；本次已有合适的 raw 输入，可直接重编码。

资源建议：

- 使用 `/home/qsy/anaconda3/envs/tlm/bin/python`，与此前回归验证环境一致。
- 单个构建进程，fast tokenizer 内部先限制为 8 个线程；现有重建入口没有 `--num_proc` 参数。
- 原始加载、编码中间缓存和最终 Arrow 会同时占用空间，建议预留 100 GiB 作为保守操作预算。这不是实测的最终数据大小。
- 当前源目录所在磁盘可用约 4.84 TiB、机器可用内存约 236 GiB，为本轮读取时的快照；正式执行前重新检查即可。
- 不把约 228 万条 JSON 一次性读入 Python list。现有构建器使用磁盘 Arrow 缓存，但仍需观察进程内存和磁盘增长。
- 保留日志、源文件摘要、tokenizer 文件摘要和代码摘要；耗时以实际构建日志为准，目前没有全量重编码耗时测量。
- 使用 `nohup + setsid` 脱离登录终端，标准输入关闭，标准输出和错误直接写日志；不依赖前台终端或 `tee` 管道存活。
- 使用文件锁阻止同一 run 重复执行，记录实际 worker PID、开始/结束时间、状态与退出码。构建期间保持输入和编码源码固定。

采用以下目录结构，将正在构建的数据与已验收的数据区分开：

```text
/home/qsy/workspace/gen_data/4090_gen_qwen_v3_boundary_len2048_20261021_01/
  run_rebuild.sh              # 独立后台 worker
  rebuild.pid                 # worker PID，不是 Python 子进程 PID
  run.lock                    # 运行期间持有的文件锁
  status                      # RUNNING / SUCCEEDED / FAILED
  exit_code                   # 正常退出或可捕获失败时写入
  started_at
  finished_at
  logs/
    rebuild.log
  manifest/
    inputs.sha256
    tokenizer.sha256
    code.sha256
  dataset.pending/             # 构建和验收期间
    .tokenize_cache/
    tokenization_stats.json
    dataset_dict.json
    train/
    validation/
  dataset/                     # 验收后由 dataset.pending 重命名得到
```

`dataset.pending` 与 `dataset` 是同一份数据的两个阶段，不是要求存两份最终 Arrow。run 目录名称仅为示例；已有同名运行时换新编号。

## 5. 后台启动、监控与结束判定

以下命令只负责提交后台任务，返回命令行后可退出 SSH。选择全新的 run 目录；创建目录失败时启动脚本立即退出，防止复用历史运行。`logs`、`manifest` 和进程状态放在输出数据目录之外，不会触发重建入口对非空目录的拒绝。

```bash
bash <<'REBUILD_LAUNCH'
set -euo pipefail

command -v nohup >/dev/null
command -v setsid >/dev/null
command -v flock >/dev/null

REBUILD_RUN_DIR=/home/qsy/workspace/gen_data/4090_gen_qwen_v3_boundary_len2048_20261021_01
mkdir "$REBUILD_RUN_DIR"
mkdir "$REBUILD_RUN_DIR/logs" "$REBUILD_RUN_DIR/manifest"

cat > "$REBUILD_RUN_DIR/run_rebuild.sh" <<'REBUILD_WORKER'
#!/usr/bin/env bash
set -euo pipefail

REBUILD_RUN_DIR=$1
REBUILD_RAW_DIR=/home/qsy/workspace/gen_data/4090_gen_qwen_v2_ext
REBUILD_TOKENIZER_DIR=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2

# Duplicate workers must not overwrite the active worker's PID or status.
exec 9>"$REBUILD_RUN_DIR/run.lock"
flock -n 9 || exit 75
if [ -e "$REBUILD_RUN_DIR/status" ]; then
  echo "This run was already started; choose a new run directory."
  exit 73
fi

write_status() {
  printf '%s\n' "$1" > "$REBUILD_RUN_DIR/status.tmp"
  mv -f "$REBUILD_RUN_DIR/status.tmp" "$REBUILD_RUN_DIR/status"
}

finish_rebuild() {
  rebuild_rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$rebuild_rc" > "$REBUILD_RUN_DIR/exit_code"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$REBUILD_RUN_DIR/finished_at"
  if [ "$rebuild_rc" -eq 0 ]; then
    write_status SUCCEEDED
  else
    write_status FAILED
  fi
  exit "$rebuild_rc"
}

trap finish_rebuild EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "$$" > "$REBUILD_RUN_DIR/rebuild.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$REBUILD_RUN_DIR/started_at"
write_status RUNNING

cd /data3/qsy/complier/llm_compiler
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8

sha256sum "$REBUILD_RAW_DIR/train_raw.json" "$REBUILD_RAW_DIR/validation_raw.json" \
  > "$REBUILD_RUN_DIR/manifest/inputs.sha256"

sha256sum "$REBUILD_TOKENIZER_DIR/tokenizer.json" \
  "$REBUILD_TOKENIZER_DIR/tokenizer_config.json" \
  "$REBUILD_TOKENIZER_DIR/config.json" \
  > "$REBUILD_RUN_DIR/manifest/tokenizer.sha256"

sha256sum LLM/data/rebuild_clm_dataset.py LLM/data/make_dataset_utils.py \
  LLM/core/tokenization_protocol.py \
  > "$REBUILD_RUN_DIR/manifest/code.sha256"

/home/qsy/anaconda3/envs/tlm/bin/python -u LLM/__main__.py rebuild-dataset \
  --train-file "$REBUILD_RAW_DIR/train_raw.json" \
  --validation-file "$REBUILD_RAW_DIR/validation_raw.json" \
  --tokenizer-path "$REBUILD_TOKENIZER_DIR" \
  --output-dir "$REBUILD_RUN_DIR/dataset.pending" \
  --max-length 2048 \
  --overlength-policy error \
  --ppt-marker PPT
REBUILD_WORKER

nohup setsid /bin/bash "$REBUILD_RUN_DIR/run_rebuild.sh" "$REBUILD_RUN_DIR" \
  </dev/null >"$REBUILD_RUN_DIR/logs/rebuild.log" 2>&1 &

printf 'Background launch submitted. Run directory: %s\n' "$REBUILD_RUN_DIR"
REBUILD_LAUNCH
```

Python 使用 `-u`，及时写出日志；worker 的 `set -e` 将数据加载、摘要计算或编码失败传播为非零退出码，EXIT trap 记录最终结果。启动命令提交成功不等于数据构建成功，需要按下述方式检查。

重新登录后，无需恢复原终端，直接查看状态和日志：

```bash
REBUILD_RUN_DIR=/home/qsy/workspace/gen_data/4090_gen_qwen_v3_boundary_len2048_20261021_01

cat "$REBUILD_RUN_DIR/status"
ps -p "$(cat "$REBUILD_RUN_DIR/rebuild.pid")" -o pid,ppid,sid,etime,stat,args
tail -n 50 "$REBUILD_RUN_DIR/logs/rebuild.log"
```

如需持续观察，单独执行：

```bash
tail -F "$REBUILD_RUN_DIR/logs/rebuild.log"
```

在该日志窗口按 Ctrl+C 只结束 `tail`，不会停止后台重建。worker 等待 Python 子进程时 CPU 占用低属于正常情况，不能据此认定任务停滞。

结束判定：

| 状态 | 含义与下一步 |
| --- | --- |
| `RUNNING` | worker 已启动；结合 PID、文件锁和日志判断进程是否仍在运行 |
| `SUCCEEDED` 且 `exit_code=0` | 构建命令完成，可以进入第 6 节数据验收；还不能直接视作数据已验收 |
| `FAILED` 且非零退出码 | 查看日志和审计报告，保留本轮目录，修正问题后换新 run 编号 |
| worker 已不存在，但状态仍为 `RUNNING` 或缺少退出码 | 任务异常中止或状态未完整写出；按未完成处理，不启动训练 |

需要排查旧 PID 被复用时，可以检查 worker 命令行是否指向本轮 `run_rebuild.sh`，并检查文件锁：

```bash
flock -n -E 75 "$REBUILD_RUN_DIR/run.lock" true
```

该检查返回 75 表示锁被占用，返回 0 表示当前未被占用。文件锁本身的文件即使存在，也不代表任务还活着。

后台隔离保证终端退出、SSH 断连和日志窗口关闭不会向任务传递终端挂断。主机重启、断电、OOM/SIGKILL 或系统强制清理进程仍可能终止任务；当前重建器不具备跨重启自动续建能力。因此不设置自动循环重启，避免失败后反复写入非空目录。

重建入口拒绝覆盖非空数据目录。失败重跑时用新的 run 编号；现有入口没有恢复非空构建目录的 CLI。目录内虽有 Hugging Face 缓存，也不能据此承诺命令可以断点续建。构建器会在最终保存前写统计报告，因此仅看到 `status=ready` 不代表 Arrow 已保存完毕。

## 6. 数据验收

下述固定数量断言仅验收原样重编码版本的 token 完整性，不验证 TVM 可构建性。正式过滤版应另外检查每条样本的验证键与通过清单，并改用该清单的 split/workload 数量。

第一层是全量数值和元数据检查。确认后台构建成功后，从任意新登录的 shell 执行：

```bash
cd /data3/qsy/complier/llm_compiler
REBUILD_RUN_DIR=/home/qsy/workspace/gen_data/4090_gen_qwen_v3_boundary_len2048_20261021_01

REBUILD_CHECK_DIR="$REBUILD_RUN_DIR/dataset.pending" \
PYTHONDONTWRITEBYTECODE=1 \
/home/qsy/anaconda3/envs/tlm/bin/python - <<'PY'
import json
import os
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path("LLM").resolve()))
from make_dataset_utils import validate_clm_dataset_integrity
from tokenization_protocol import PPT_BOUNDARY_VERSION

root = Path(os.environ["REBUILD_CHECK_DIR"])
assert (root.parent / "status").read_text().strip() == "SUCCEEDED"
assert (root.parent / "exit_code").read_text().strip() == "0"
assert (root / "dataset_dict.json").is_file(), "Final Arrow save has not completed"
stats = json.loads((root / "tokenization_stats.json").read_text())
assert stats["status"] == "ready"
assert stats["max_length"] == 2048
assert stats["overlength_policy"] == "error"
assert stats["truncation"] is False
assert stats["ppt_boundary_version"] == PPT_BOUNDARY_VERSION

data = load_from_disk(str(root), keep_in_memory=False)
expected = {"train": (2169636, 2061), "validation": (111611, 108)}
assert set(data) == set(expected)
for split, (rows, workloads) in expected.items():
    dataset = data[split]
    audit = stats["splits"][split]
    validate_clm_dataset_integrity(dataset, split)
    assert len(dataset) == audit["input_records"] == audit["saved_records"] == rows
    assert audit["workload_count"] == workloads
    assert audit["overlength_records"] == audit["dropped_overlength_records"] == 0
    assert audit["missing_ppt_records"] == 0
    assert audit["original_length"]["max"] <= 2048
    assert audit["suffix_length"]["min"] >= 16
    assert not audit["workloads_without_records_within_limit"]
    for batch in dataset.select_columns(["length", "ppt_end"]).iter(batch_size=10000):
        assert all(
            0 <= end < length and length - end - 1 >= 16
            for length, end in zip(batch["length"], batch["ppt_end"])
        )
    print(split, "passed", rows, "rows; max_length", audit["original_length"]["max"])
print("Full metadata checks passed")
PY
```

第二层是原文对照和训练衔接检查：

| 验收项 | 方法与通过标准 |
| --- | --- |
| 全文 token 一致 | 使用相同 tokenizer，对每个 workload 至少取首条及长样本，再加固定种子的随机样本；与 Arrow 同 split、同行号的 `input_ids` 完整比较，不能只比较长度 |
| labels 内容 | 对照样本的保存态 `labels == input_ids`；padding 和 prompt mask 由训练 collator 添加 |
| PPT 边界 | 使用共享 marker 函数重新定位，结果与 `ppt_end` 相同；marker 后的分隔空格进入监督 suffix |
| 长序列尾部 | 对超过 1024 的样本重点检查末尾 TBS/PRS 等原始决策 token，确认与原文完整编码相同 |
| vocabulary | 样本 token id 在训练模型 embedding 范围内，训练与生成均使用固定 tokenizer |
| workload 划分 | 保留原始 canonical workload/source_file 分组核查结果；正式 Arrow 会移除这些身份字段，不能只看最终 Arrow 判断是否泄漏 |
| GPU smoke | 发布后衔接第 7 节，在新数据上以每卡 batch=1、短步数和含长样本的 batch 验证训练、评估与保存，记录峰值显存及吞吐 |

`sequence_complete=True` 只说明输入文本的 token 未被截断。TVM state 恢复、编译和真实延迟要由原始可恢复调度记录及后续生成实验另行验证，不能从该标记推出调度合法或高性能。

完成数据完整性和原文对照验收后发布目录，目标 `dataset` 必须不存在：

```bash
test ! -e "$REBUILD_RUN_DIR/dataset" && \
  mv -T "$REBUILD_RUN_DIR/dataset.pending" "$REBUILD_RUN_DIR/dataset"
```

发布后再 `load_from_disk` 检查一次可读性；训练配置的 `DATASET_NAME` 指向 `dataset`。`.tokenize_cache` 是中间缓存，可能占用较多空间；等新目录完成重载和训练 smoke 后再单独考虑清理，不在构建或验收过程中删除。

## 7. 与继续训练衔接

新数据通过验收后，先从现有 Stage1 checkpoint 进行 Stage2 后缀监督对照；不用旧 LAPO checkpoint 作为这一轮结构修复对照的起点。新运行使用新的模型输出目录，避免自动恢复旧优化器状态。

必须显式设置：

```text
TRAIN_STAGE=stage2
MODEL_NAME_OR_PATH=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2
TOKENIZER_NAME=/home/qsy/huggingface/model/Qwen3-0.6B-4090-struct-stage1-v2-ext-b2
DATASET_NAME=<本轮 run 目录>/dataset
ALLOW_LEGACY_DATASET=0
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_CHECKPOINTING=1
```

使用 `run_train_qwen3_clm.py` 时从 `LLM/train` 目录启动，因为该 CLM launcher 仍通过相对路径调用训练脚本；显式指定 GPU、日志和输出路径。先使用 `SMOKE_TEST=1` 完成 20 步左右的小样本检查，正式 batch 和梯度累积在显存实测后确定。随机 smoke 子集不一定覆盖最长样本，需要另确保至少一个长序列 batch 被执行。

训练后的主要验收目标是固定 workload、sketch、target 和生成预算下，超过 1024 的旧失败 workload 能否产生完整 state，以及编译通过率是否提高。偏好准确率或 token accuracy 仅作为辅助指标。

## 8. 执行顺序与交付物

1. 确定产物用途：原样重编码对照可使用本文命令；正式可构建训练集须先完成 [源调度审计与过滤](05_dump_dataset_validity_analysis.md)，该审计已完成，正式重建执行记录见该文第 9.9 节。随后固定输入、tokenizer 和当前代码摘要；保留原始划分与 hold-out 隔离。
2. 通过 `nohup + setsid` 提交独立后台重编码任务，写入新的 `dataset.pending`，保存完整日志、PID、状态和退出码；退出 SSH 后仍继续运行。
3. 检查全量统计、完整性和代表样本原文对照；所有样本保留，workload 覆盖保持一致。
4. 将通过数据验收的目录改名为 `dataset` 并重新加载，随后完成新数据的训练 smoke。
5. 训练 smoke 通过后启动独立 Stage2 继续训练对照，再评估生成有效率、编译通过率与真实性能。

应交付可重载的 DatasetDict、`tokenization_stats.json`、后台 worker 脚本及运行状态记录、源数据/tokenizer/代码摘要、验收结果、训练 smoke 日志，以及旧模型与新模型的固定预算生成对照。全量重建本身不能作为生成问题已经解决的结论。
