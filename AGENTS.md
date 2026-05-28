# Repository Guidelines

## Project Structure & Module Organization
This repository extends Apache TVM with TLM/LLM-based compilation utilities. Core C++ compiler/runtime code lives in `src/` with public headers in `include/`. Python bindings are under `python/tvm/`. TLM data generation, training, measurement, and tuning scripts are in `gen/`; planning notes are in `plan/`. Tests are grouped in `tests/python/`, `tests/cpp/`, `tests/micro/`, and `tests/lint/`. Documentation and tutorials live in `docs/` and `gallery/`. Build helpers are in `cmake/`, `configs/`, `docker/`, and `conda/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies for TLM training and scripts.
- `mkdir -p build && cp cmake/config.cmake build/config.cmake`: create a local build config; edit it for LLVM, CUDA, or other backends.
- `cmake .. && make -j4` from `build/`: configure and compile TVM libraries.
- `make cython3`: rebuild in-place Python extension modules after API changes.
- `make lint`, `make pylint`, `make cpplint`, `make mypy`: run project lint/type checks.
- `python gen/dump_programs.py --target=nvidia/nvidia-v100`: example TLM data-generation entry point.

## Coding Style & Naming Conventions
Python uses Black with 100-character lines, configured in `pyproject.toml`; keep modules and functions `snake_case`. C++ follows Google-style `clang-format` with 100-character columns and left pointer alignment; mirror naming in nearby TVM files. Rust code under `rust/` should pass `cargo fmt --all`. Keep generated datasets, build products, and local model outputs out of source control unless explicitly required.

## Testing Guidelines
Use the narrowest relevant test first. Python tests use pytest-style files under `tests/python/`, commonly named `test_*.py`; for example, run `python -m pytest tests/python/unittest/test_target_codegen_llvm.py`. C++ and runtime tests use `make cpptest`, `make crttest`, or task scripts in `tests/scripts/`. For `gen/`, validate with a small target or reduced dataset before long GPU measurements.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `add baseline.py` or `improve training time`. Keep the first line concise and user-visible. Pull requests should include a problem statement, summary, commands/tests run, affected targets or hardware, and links to issues or experiment artifacts when relevant. Include screenshots only for documentation or web UI changes.

## Security & Configuration Tips
Do not commit downloaded datasets, credentials, SSH host details, or machine-specific paths. Keep local backend settings in `build/config.cmake`. For CUDA or measurement runs, document GPU model, target string, and exclusivity assumptions.
