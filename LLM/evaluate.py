# import matplotlib.pyplot as plt

# # 数据准备
# categories = ['base', 'optimized', 'Ansor']
# values = [9.6761, 8.4343, 4.1232, 3.9473, 3.8342, 3.5221]

# # 创建图形
# plt.figure(figsize=(6, 5))  # 优化高度比例

# # 绘制柱状图
# bars = plt.bar(categories, values, 
#                color=['skyblue', 'green', 'red'], 
#                edgecolor='black',
#                width=0.6)  # 适当减小宽度使标签更紧凑

# # 添加标签和标题
# plt.xlabel('Model Variant', fontsize=12, labelpad=8)
# plt.ylabel('Inference Latency (ms)', fontsize=12, labelpad=8)
# plt.title('resnet50 & bert_base Inference Evaluation', fontsize=14, pad=12)

# # 关键修改：缩短标签距离
# for bar in bars:
#     height = bar.get_height()
#     # 垂直偏移量从1改为0.15（单位：数据单位）
#     # 添加边框提高可读性
#     plt.text(bar.get_x() + bar.get_width()/2, 
#              height + 0.15,  # 缩短距离的核心参数
#              f'{height:.4f}',  # 保留4位小数
#              ha='center', 
#              va='bottom',
#              fontsize=11,
#              bbox=dict(facecolor='white',  # 白色背景
#                        alpha=0.7,
#                        edgecolor='none',
#                        boxstyle='round,pad=0.2'))  # 圆角边框

# # 优化Y轴范围（减少顶部空白）
# max_val = max(values)
# plt.ylim(0, max_val * 1.15)  # 保留15%顶部空间

# # 网格线增强可读性
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # 优化布局
# plt.tight_layout(pad=2.0)  # 增加内边距
# plt.savefig('bar_chart_2.png', dpi=300, bbox_inches='tight')  # 高清保存

import matplotlib.pyplot as plt
import numpy as np

# 数据准备 - 按模型分组
models = ['resnet50', 'bert_base']
variants = ['base', 'optimized', 'Ansor']

# 每个模型对应三个变体的延迟数据
resnet50_data = [8.4343, 4.1232, 3.8342]
bert_base_data = [9.6761, 3.9473, 3.5221]

# 创建图形
plt.figure(figsize=(10, 6))

# 设置柱状图参数
bar_width = 0.35
x = np.arange(len(variants))  # 变体位置

# 绘制分组柱状图
resnet_bars = plt.bar(x - bar_width/2, resnet50_data, 
                     bar_width, color='skyblue', edgecolor='black', label='resnet50')
bert_bars = plt.bar(x + bar_width/2, bert_base_data, 
                   bar_width, color='lightcoral', edgecolor='black', label='bert_base')

# 添加数据标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, 
                 height + 0.1,  # 动态调整标签位置
                 f'{height:.4f}', 
                 ha='center', 
                 va='bottom',
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, 
                           edgecolor='none', boxstyle='round,pad=0.2'))

add_labels(resnet_bars)
add_labels(bert_bars)

# 添加标签和标题
plt.xlabel('Model Variant', fontsize=12, labelpad=10)
plt.ylabel('Inference Latency (ms)', fontsize=12, labelpad=10)
plt.title('resnet50 & bert_base Inference Evaluation', fontsize=15, pad=15)

# 设置x轴刻度
plt.xticks(x, variants, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
plt.legend(title='Model', loc='upper right', fontsize=10)

# 优化Y轴范围（自动适应最大值）
max_val = max(max(resnet50_data), max(bert_base_data))
plt.ylim(0, max_val * 1.15)

# 优化布局
plt.tight_layout(pad=2.0)
plt.savefig('combined_bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()