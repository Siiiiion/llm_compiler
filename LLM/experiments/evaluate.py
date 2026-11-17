"""Plot the historical illustrative latency values; not a benchmark command."""

import argparse


def add_labels(bars):
    import matplotlib.pyplot as plt

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,  # 动态调整标签位置
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="combined_bar_chart.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    # 数据准备 - 按模型分组
    variants = ["base", "optimized", "Ansor"]

    # 每个模型对应三个变体的延迟数据
    resnet50_data = [8.4343, 4.1232, 3.8342]
    bert_base_data = [9.6761, 3.9473, 3.5221]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 设置柱状图参数
    bar_width = 0.35
    x = np.arange(len(variants))  # 变体位置

    # 绘制分组柱状图
    resnet_bars = plt.bar(
        x - bar_width / 2,
        resnet50_data,
        bar_width,
        color="skyblue",
        edgecolor="black",
        label="resnet50",
    )
    bert_bars = plt.bar(
        x + bar_width / 2,
        bert_base_data,
        bar_width,
        color="lightcoral",
        edgecolor="black",
        label="bert_base",
    )

    # 添加数据标签

    add_labels(resnet_bars)
    add_labels(bert_bars)

    # 添加标签和标题
    plt.xlabel("Model Variant", fontsize=12, labelpad=10)
    plt.ylabel("Inference Latency (ms)", fontsize=12, labelpad=10)
    plt.title("resnet50 & bert_base Inference Evaluation", fontsize=15, pad=15)

    # 设置x轴刻度
    plt.xticks(x, variants, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 添加图例
    plt.legend(title="Model", loc="upper right", fontsize=10)

    # 优化Y轴范围（自动适应最大值）
    max_val = max(max(resnet50_data), max(bert_base_data))
    plt.ylim(0, max_val * 1.15)

    # 优化布局
    plt.tight_layout(pad=2.0)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
