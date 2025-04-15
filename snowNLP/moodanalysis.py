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
        # 保存情感指数到文件
        with open('./output/sentiment.txt', 'a', encoding='utf-8') as file:
            file.write(f"文本{i}: {sentence}\n")
            file.write(f"情感指数: {sentiment:.4f}\n")
            # 写入平均值
            file.write(f"平均情感指数: {sum(sentiments) / len(sentiments):.4f}\n")
            file.close()
    return sentiments


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


if __name__ == "__main__":
    main()