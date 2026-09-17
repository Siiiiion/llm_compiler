# Repository Guidelines

## Project Structure & Module Organization

This repository extends Apache TVM with language-model-guided tensor scheduling.

- `src/`, `include/`, `python/tvm/`: C++ compiler/runtime, public headers, and Python bindings.
- `LLM/`: Qwen3 schedule generation, measurement, and Relay benchmarking; `LLM/train/`: pretraining and DPO/LAPO. `gen/` retains original TLM workflows.
- `tests/python/`, `tests/cpp/`, `tests/micro/`, `tests/lint/`: tests and quality checks.
- `docs/`, `gallery/`: documentation and tutorials; `plan/`: design notes. Verify plans against implementation.

## Build, Test, and Development Commands

Run from the repository root unless indicated. For a first build, copy `cmake/config.cmake` to `build/config.cmake` and configure LLVM/CUDA.

- `pip install -r requirements.txt`: install legacy TLM dependencies. For Qwen3, consult `LLM/train/qwen3_train_guide.md` and verify model-library compatibility.
- `cmake -S . -B build && cmake --build build -j4`: configure and build TVM.
- `pip install -e ./python`: install local Python bindings; `make cython3`: rebuild extensions.
- `make lint`, `make mypy`: run lint and type checks; `make pylint` and `make cpplint` select individual checks.
- `cd LLM && python gen_state.py --help`: inspect generation options before selecting checkpoints and workload registries.

## Coding Style & Naming Conventions

Python uses four-space indentation, `snake_case`, and Black with 100-character lines (`pyproject.toml`). C++ follows Google `clang-format`: two-space indentation, 100 columns, and left pointer alignment (`.clang-format`). Format changed files only; for Rust, run `cargo fmt --all` from `rust/`.

## Testing Guidelines

Add focused regression tests for behavior changes. Python uses pytest and `test_*.py`; run, for example, `python -m pytest tests/python/unittest/test_target_codegen_llvm.py`. Build C++ GoogleTest tests with `make cpptest`, then execute `./build/cpptest`.

For `LLM/` and `gen/`, smoke-test reduced inputs through generation, build validation, and measurement. Performance comparisons require matching workloads/targets, output correctness, and complete schedule history. The Relay baseline uses `opt_level=3`.

## Commit & Pull Request Guidelines

Use concise imperative summaries, following local history: `add qwen3 workloads`, `improve training time`. PRs should explain the problem, behavior changes, validation commands/results, affected hardware, and relevant issues or artifacts. Include screenshots for visual changes.

## Security & Configuration Tips

Keep datasets, checkpoints, build outputs, credentials, and machine-specific paths out of commits. Check `LLM/common.py` for path and hold-out routing. Record checkpoint/tokenizer paths, GPU, target, timing settings, and output locations; measure on an exclusive GPU.
