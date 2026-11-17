"""Generate the architecture module index from current Python definitions and imports."""

import argparse
import ast
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parents[1]
GROUPS = (
    "core",
    "data",
    "generation",
    "audit",
    "validation",
    "train",
    "benchmark",
    "experiments",
    "attention",
    "export_onnx",
)


def render():
    lines = [
        "# 模块与函数索引",
        "",
        "从当前 Python AST 生成。根目录只有包标识和统一入口；下表统计各子包的实现。",
        "“依赖”列只列包内静态导入，包含函数内延迟导入；不表示运行时会全部加载。函数列包含模块顶层函数和类，类内部方法请进入源码查看。",
        "",
        "生成命令：python LLM/tools/update_module_index.py；校验：追加 --check。",
    ]
    for group in GROUPS:
        lines += [
            "",
            "## " + group,
            "",
            "| 实现 | 行数 | 函数 / 类 | 包内依赖 |",
            "| --- | ---: | --- | --- |",
        ]
        for path in sorted((LLM_ROOT / group).glob("*.py")):
            if path.name == "__init__.py":
                continue
            source = path.read_text()
            tree = ast.parse(source)
            names = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            deps = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("LLM."):
                    deps.add(node.module.removeprefix("LLM."))
                elif isinstance(node, ast.Import):
                    deps.update(
                        item.name.removeprefix("LLM.")
                        for item in node.names
                        if item.name.startswith("LLM.")
                    )
            lines.append(
                f"| [{path.name}](../{group}/{path.name}) | {len(source.splitlines())} | {', '.join(names) or '常量/入口'} | {', '.join(sorted(deps)) or '无'} |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = LLM_ROOT / "docs/MODULE_INDEX.md"
    text = render()
    if args.check:
        if not destination.exists() or destination.read_text() != text:
            parser.exit(1, "Module index is stale; run without --check.\n")
    else:
        destination.write_text(text)


if __name__ == "__main__":
    main()
