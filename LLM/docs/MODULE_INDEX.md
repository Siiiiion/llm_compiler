# 模块与函数索引

从当前 Python AST 生成。根目录只有包标识和统一入口；下表统计各子包的实现。
“依赖”列只列包内静态导入，包含函数内延迟导入；不表示运行时会全部加载。函数列包含模块顶层函数和类，类内部方法请进入源码查看。

生成命令：python LLM/tools/update_module_index.py；校验：追加 --check。

## core

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [common.py](../core/common.py) | 264 | clean_name, register_data_path, get_relay_ir_filename, get_task_info_filename, load_tasks_path, load_and_register_tasks, get_to_measure_filename, get_measure_record_filename, hold_out_task_files, hold_out_task_files_by_workloads, yield_hold_out_five_files, get_hold_out_five_files, get_bert_files, get_hold_out_files_by_workloads | 无 |
| [io.py](../core/io.py) | 63 | file_sha256, atomic_json, canonical_workload_key, atomic_write_text | 无 |
| [layout.py](../core/layout.py) | 60 | source_path, module_command, project_pythonpath, code_dependencies | 无 |
| [runtime.py](../core/runtime.py) | 24 | now, bounded_pool | 无 |
| [schedule_record_utils.py](../core/schedule_record_utils.py) | 126 | canonical_json, digest, workload_key, input_hash, normalize_record, measured_latency, read_journal, append_journal, classify_build_error | core.io |
| [tokenization_protocol.py](../core/tokenization_protocol.py) | 80 | build_ppt_marker_id_candidates, find_ppt_marker_end, encode_generation_prompts, encode_prompt_suffix | 无 |

## data

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [extend_tokenizer.py](../data/extend_tokenizer.py) | 379 | _load_extra_tokens_file, _dedup_preserve_order, _subword_ids, _init_from_subwords, parse_args, collect_candidate_tokens, main | 无 |
| [make_dataset.py](../data/make_dataset.py) | 1092 | _build_logger, ScriptArguments, for_clm_or_mlm, _structure_to_text, _normalize_measure_record, _get_latency, _merge_stats, _split_files_for_validation, _prepare_dataset_output_dir, _concat_json_files, _parse_eval_workloads, for_gen_basic, for_gen, for_gen_best, softmax, for_gen_eval_sketch, input_to_tokens, process_file, token_files_and_merge, main | core.common, core.schedule_record_utils, data.make_dataset_utils, data.tokenizer, generation.task_scheduler |
| [make_dataset_utils.py](../data/make_dataset_utils.py) | 467 | _resolve_max_length, json_dfs_without_bracket, json_dfs_with_bracket, json_to_token, _load_raw_datasets, _length_summary, _clm_length_stats, validate_clm_dataset_integrity, make_dataset, make_dataset_test | core.tokenization_protocol |
| [prepare_audited_clm.py](../data/prepare_audited_clm.py) | 183 | load_manifest, load_verified_source, prepare_raw, main | audit.audit_tvm_backend, core.io, core.layout, core.schedule_record_utils, data.make_dataset |
| [rebuild_audited_clm.py](../data/rebuild_audited_clm.py) | 356 | snapshot_inputs, verify_snapshot, reuse_completed_stages, run, main | audit.audit_tvm_backend, core.io, core.layout, core.runtime, data.prepare_audited_clm, data.verify_audited_clm |
| [rebuild_clm_dataset.py](../data/rebuild_clm_dataset.py) | 103 | verify_build_audit, main | core.io, data.make_dataset_utils, data.prepare_audited_clm |
| [sampling.py](../data/sampling.py) | 35 | length_bucket, operator_family, representative_indices | 无 |
| [split_measure_records.py](../data/split_measure_records.py) | 142 | _canonical_workload_key, _target_kind, _output_name, split_records, main | core.common |
| [tokenizer.py](../data/tokenizer.py) | 99 | train_tokenizer, test_model_max_length | 无 |
| [verify_audited_clm.py](../data/verify_audited_clm.py) | 224 | require, verify_dataset, main | core.io, core.schedule_record_utils, core.tokenization_protocol, data.make_dataset_utils, data.prepare_audited_clm, data.rebuild_clm_dataset |

## generation

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [constrained_generation.py](../generation/constrained_generation.py) | 467 | ConstraintError, verify_gpu_resources, divisors, SplitSpec, resource_groups, DecisionPlan, CompletionPool, resource_plan, TokenGrammar, generate_constrained | core.schedule_record_utils, generation.resource_feasibility |
| [gen_state.py](../generation/gen_state.py) | 814 | ScriptArguments, load_tokenizer, load_model, _resolve_max_new_tokens, gen_func, worker, _atomic_write_json, _atomic_merge_parts, _completed_workloads, _make_stats_payload, main | core.common, core.layout, data.make_dataset, generation.constrained_generation, generation.generation_protocol, generation.measured_history |
| [generation_protocol.py](../generation/generation_protocol.py) | 124 | resolve_max_new_tokens, decode_decision_tokens, isolated_generation_rng, generate_decisions | core.tokenization_protocol, data.make_dataset, generation.constrained_generation |
| [measured_history.py](../generation/measured_history.py) | 16 | check_measured | core |
| [resource_feasibility.py](../generation/resource_feasibility.py) | 142 | _hardware_limits, _completed_steps, _bind_tiled_roots, _shared_allocations, preview_ppt_resources | 无 |
| [task_scheduler.py](../generation/task_scheduler.py) | 319 | ScriptArguments, _valid_latency, read_fine_tuning_json, _count_measurement_attempts, _expand_history_files, _load_task_files, _score_history, _default_output_path, _load_schedule_payload, find_potential_files, find_potential_files_len, main | core.common |

## audit

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [audit_persistent_builder.py](../audit/audit_persistent_builder.py) | 68 | _build_default, PersistentLocalBuilder | 无 |
| [audit_reuse.py](../audit/audit_reuse.py) | 120 | compatible_origin, reuse_results | core.io, core.layout, core.schedule_record_utils, data.prepare_audited_clm |
| [audit_schedule_dataset.py](../audit/audit_schedule_dataset.py) | 596 | source_snapshot, scan_raw_splits, discover_sources, make_manifest, audit_source, run, parse_args, main | audit.audit_reuse, audit.audit_tvm_backend, core.common, core.io, core.layout, core.runtime, core.schedule_record_utils |
| [audit_tvm_backend.py](../audit/audit_tvm_backend.py) | 361 | compiler_fingerprint, register_tasks, _prepare, TVMAuditor | audit.audit_persistent_builder, core.io, core.schedule_record_utils |

## validation

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [candidate_metrics.py](../validation/candidate_metrics.py) | 89 | summarize_workload, aggregate | 无 |
| [eval_struct.py](../validation/eval_struct.py) | 620 | _evaluate_on_primary, _require_tvm, _setup_target_and_tasks, _read_sketch_workloads, _resolve_max_new_tokens, _generate_decisions_for_states, evaluate_structural_validity, StructuralEvalCallback, CliArgs, _load_model_tokenizer, main | core.common, generation.constrained_generation, generation.generation_protocol |
| [generation_roundtrip_worker.py](../validation/generation_roundtrip_worker.py) | 267 | initialize, replay_suffix, verify_full_text, StageProgress, build_candidate, check_sample | audit.audit_tvm_backend, core.io, core.schedule_record_utils, core.tokenization_protocol, data.make_dataset, generation.gen_state |
| [multisketch_metrics.py](../validation/multisketch_metrics.py) | 105 | summarize_multisketch | validation.candidate_metrics |
| [multisketch_selection.py](../validation/multisketch_selection.py) | 272 | sketch_identity, _require, select_multisketch_suite | core.io, core.schedule_record_utils, data.prepare_audited_clm |
| [validate_constrained_generation.py](../validation/validate_constrained_generation.py) | 449 | check_worker, select_samples, run_case, run | audit.audit_tvm_backend, core.io, core.layout, core.schedule_record_utils, data.make_dataset, experiments.diagnose_model_generation, generation.constrained_generation, generation.gen_state, generation.generation_protocol, validation.eval_struct |
| [validate_generation_roundtrip.py](../validation/validate_generation_roundtrip.py) | 504 | input_snapshot, verify_inputs, select_samples, summarize, run, main | core.io, core.layout, core.runtime, core.schedule_record_utils, data.prepare_audited_clm, data.rebuild_audited_clm, data.rebuild_clm_dataset, data.sampling, validation.generation_roundtrip_worker |
| [validate_multicandidate_generation.py](../validation/validate_multicandidate_generation.py) | 449 | verify_inputs, read_worker_check, inspect_candidate, run_attempt, run_shard, prepare, run | audit.audit_tvm_backend, core.io, core.layout, core.schedule_record_utils, data.make_dataset, experiments.diagnose_model_generation, generation.gen_state, generation.generation_protocol, validation.candidate_metrics, validation.eval_struct |
| [validate_multisketch_generation.py](../validation/validate_multisketch_generation.py) | 322 | mode_order, run_shard, prepare, run | audit.audit_tvm_backend, core.io, core.layout, generation.gen_state, validation.candidate_metrics, validation.multisketch_metrics, validation.multisketch_selection, validation.validate_multicandidate_generation |

## train

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [build_preference_pairs.py](../train/build_preference_pairs.py) | 492 | PreferenceArgs, _build_logger, _dfs_flatten, _compute_weight, _normalize_record, _get_latency, _process_file, _split_files_for_validation, _merge_stats, _run_group, main | core.common, core.schedule_record_utils, data.make_dataset_utils |
| [run_train_qwen3_clm.py](../train/run_train_qwen3_clm.py) | 278 | _parse_cuda_visible_devices, _is_valid_local_model_dir, _is_valid_local_tokenizer_dir, _looks_like_local_path, _env_flag, _append_optional_arg, main | 无 |
| [run_train_qwen3_dpo.py](../train/run_train_qwen3_dpo.py) | 354 | _env_flag, _parse_cuda_visible_devices, _is_valid_local_model_dir, _is_valid_local_tokenizer_dir, _looks_like_local_path, _append_optional_arg, main | 无 |
| [train_qwen3_clm.py](../train/train_qwen3_clm.py) | 693 | ModelArguments, DataTrainingArguments, _find_subseq_end, PPTSuffixDataCollator, _load_tokenizer, _maybe_shuffle_and_select, _apply_ppt_suffix_mask, main | core.tokenization_protocol, data.make_dataset_utils, validation.eval_struct |
| [train_qwen3_dpo.py](../train/train_qwen3_dpo.py) | 896 | ModelArguments, DPODataArguments, DPOTrainingExtraArguments, _tokenize_pair, _load_preference_dataset, DPOCollator, _lapo_weight_statistics, _compute_sequence_logps, _dpo_loss, DPOTrainer, _load_tokenizer, _load_causal_model, main | core.tokenization_protocol |

## benchmark

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [analyze_missing_history.py](../benchmark/analyze_missing_history.py) | 533 | canonical_target, valid_latency, resolve_record_files, read_record_index, load_summary, select_variants, parse_task_filename, discover_task_files, load_selected_tasks, source_counts, build_network_reports, write_csv, build_log, main | core.io |
| [baseline.py](../benchmark/baseline.py) | 266 | _default_output_log_path, _parse_args, main | 无 |
| [benchmark_relay.py](../benchmark/benchmark_relay.py) | 708 | InputSpec, _json_object, normalize_input_specs, load_relay_artifact, load_onnx_model, hardware_params_for_target, _valid_latency, _canonical_target, load_history_stats, history_coverage, filter_history_for_build, generate_inputs, input_report, build_or_load, timing_statistics, run_library, compare_outputs, write_report, parse_args, main | 无 |
| [benchmark_relay_batch.py](../benchmark/benchmark_relay_batch.py) | 640 | parse_args, now_iso, file_fingerprint, artifact_stem, network_family, slug, discover_artifacts, discover_histories, load_json, variant_result, numeric, summary_stats, aggregate_network, write_csv, stat_value, write_network_outputs, RunLogger, main | core.io, core.layout |
| [classify_network_speedup.py](../benchmark/classify_network_speedup.py) | 445 | discover_json_files, read_best_latencies, network_family, parse_task_filename, discover_task_files, classify_status, load_registry_entries, geomean, summarize_entries, grouped_summaries, write_csv, format_value, build_log, main | core.io |
| [measure_programs.py](../benchmark/measure_programs.py) | 576 | _pre_parse_cuda_visible, make_measurer, _make_measurer_kwargs, _latency_of, _split_records, remeasure_batch, measure_file_inplace, legacy_single_file_mode, _filter_files, main | core.common |
| [tune_relay.py](../benchmark/tune_relay.py) | 379 | _parse_args, _valid_latency, _load_history_stats, _format_workload_key, _check_history_coverage, _build_with_auto_scheduler, _build_with_topi, main | 无 |

## experiments

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [diagnose_generation_interventions.py](../experiments/diagnose_generation_interventions.py) | 345 | replace_split_headers, raise_thread_factor, initial_geometry, run_case, run | audit.audit_tvm_backend, core.io, core.layout, core.schedule_record_utils, data.make_dataset, experiments.diagnose_model_generation, generation.gen_state, generation.generation_protocol, validation.eval_struct, validation.generation_roundtrip_worker |
| [diagnose_generation_roundtrip.py](../experiments/diagnose_generation_roundtrip.py) | 313 | load_source, verify_compatible_inputs, run | audit.audit_tvm_backend, core.io, core.layout, core.runtime, core.schedule_record_utils, validation.generation_roundtrip_worker, validation.validate_generation_roundtrip |
| [diagnose_model_generation.py](../experiments/diagnose_model_generation.py) | 385 | split_expectations, inspect_split_output, TracedModel, make_sketch, trace_case, run | audit.audit_tvm_backend, core.io, core.layout, core.schedule_record_utils, data.make_dataset, generation.gen_state, generation.generation_protocol, validation.eval_struct |
| [evaluate.py](../experiments/evaluate.py) | 92 | add_labels, main | 无 |
| [measure_plan6_representatives.py](../experiments/measure_plan6_representatives.py) | 191 | reference, run | audit.audit_tvm_backend, core.io, core.runtime, core.schedule_record_utils |
| [run_plan6_controlled_experiment.py](../experiments/run_plan6_controlled_experiment.py) | 300 | prepare, eval_workloads, run | audit.audit_tvm_backend, core.io, core.layout, core.runtime, core.schedule_record_utils, core.tokenization_protocol, generation.gen_state, validation.eval_struct |

## attention

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [benchmark_flash_attention.py](../attention/benchmark_flash_attention.py) | 195 | parse_args, cuda_timing, eager_attention, main | 无 |
| [benchmark_tvm_attention.py](../attention/benchmark_tvm_attention.py) | 195 | labeled_path, safe_label, parse_args, main | benchmark.benchmark_relay |
| [compare_results.py](../attention/compare_results.py) | 119 | parse_args, load_json, load_output, main | 无 |
| [create_attention_task.py](../attention/create_attention_task.py) | 330 | hardware_params_for_target, build_attention_module, sample_sketches, atomic_pickle, atomic_json, truncate_records_at_ppt, parse_args, main | 无 |
| [metrics.py](../attention/metrics.py) | 63 | timing_statistics, compare_arrays, atomic_json | 无 |
| [run_generation.py](../attention/run_generation.py) | 135 | parse_args, run, generation_command, measurement_command, main | core.layout |

## export_onnx

| 实现 | 行数 | 函数 / 类 | 包内依赖 |
| --- | ---: | --- | --- |
| [export_onnx_qwen3.py](../export_onnx/export_onnx_qwen3.py) | 349 | CausalLMLogitsWrapper, parse_args, get_torch_dtype, parse_positive_int_list, sanitize_tag, logits_mode_name, grid_output_path, export_one, write_manifest, load_model, main | 无 |
