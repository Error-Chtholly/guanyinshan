import os
from snownlp import SnowNLP
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 或 'Qt5Agg'、'Agg'等

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 设置字体大小
plt.rcParams['font.size'] = 17

# 读取文本文件
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # 按句号分割成句子，并去除空字符串
    sentences = [s.strip() for s in content.split('。') if s.strip()]
    return sentences


# 情感分析函数
def analyze_sentiment(sentences):
    sentiments = []
    for i, sentence in enumerate(sentences, 1):
        s = SnowNLP(sentence)
        sentiment = s.sentiments
        sentiments.append(sentiment)
        print(f"文本{i}: {sentence}")
        print(f"情感指数: {sentiment:.4f}\n")
    return sentiments


# 绘制情感指数图表
def plot_sentiments(sentiments, save_path=None, dpi=1080, figsize=(14, 11)):
    plt.figure(figsize=figsize, dpi=dpi)

    # 设置更大的带宽参数（关键修改）
    bandwidth = 0.41  # 原值0.1，现增大3倍

    # 生成网格数据
    x = np.arange(1, len(sentiments) + 1)
    y = np.array(sentiments)
    xi = np.linspace(1, len(sentiments), 1000)
    yi = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xi, yi)

    # 改进的高斯核计算（增加带宽影响）
    density = np.zeros_like(X)
    for i, val in enumerate(y):
        r_sq = ((X - x[i]) ** 2 + (Y - val) ** 2) / (2 * bandwidth ** 2)
        density += np.exp(-r_sq) / (2 * np.pi * bandwidth ** 2)

    # 绘制热力图（保持50%透明度）
    plt.pcolormesh(X, Y, density, cmap='coolwarm', alpha=0.5, shading='auto')

    # 保持原始折线图
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Affective index')
    avg = np.mean(sentiments)
    plt.axhline(y=avg, color='r', linestyle='--', label=f'Mean value ({avg:.4f})')

    # 调整颜色条范围
    plt.clim(0, density.max() * 0.6)  # 降低最大值显示比例使颜色更饱和

    # 其余样式保持不变
    plt.colorbar(label='Density')
    plt.xlabel('Text sequence number', fontsize=17)
    plt.ylabel('Affective index', fontsize=17)
    plt.xticks(range(1, len(sentiments) + 1, max(1, len(sentiments) // 20)))
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(bbox_to_anchor=(0.81, 0.1), loc='center')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')


# 主程序
def main():
    # 替换为你的txt文件路径
    file_path = './data/web-text.txt'  # 请确保文件是UTF-8编码

    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    # 读取并分割文本
    sentences = read_text_file(file_path)

    if len(sentences) != 241:
        print(f"警告：文本数量为 {len(sentences)}，不是预期的241条")

    # 进行情感分析
    print("开始情感分析...\n")
    sentiments = analyze_sentiment(sentences)

    # 计算并打印平均值
    avg_sentiment = sum(sentiments) / len(sentiments)
    print(f"\n所有文本的平均情感指数: {avg_sentiment:.4f}")

    # 绘制图表并保存
    plot_sentiments(sentiments,
                    save_path='./output/sentiment_analysis_heatmap.png',  # 保存路径
                    dpi=720,  # 分辨率设置(1080dpi)
                    figsize=(16, 9))  # 图片大小(16x8英寸)


if __name__ == "__main__":
    main()