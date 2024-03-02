import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def drawevalplot(scores):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    colors = ["#4682B4","#FF8F00"]

    # 误差线使用标准偏差SD 方差的平方根
    models = ["0", "1", "2", "3", "4"]
    qlabels = ['Plurality', 'Definiteness']

    # 数据处理
    qa_label_result_means = {}
    qa_label_result_sds = {}
    for model in scores.keys():
        for label in scores[model].keys():
            if label not in qa_label_result_means.keys():
                qa_label_result_means[label] = []
            if label not in qa_label_result_sds.keys():
                qa_label_result_sds[label] = []
            temp = np.array(scores[model][label])
            qa_label_result_means[label].append(np.mean(temp))
            qa_label_result_sds[label].append(sqrt(np.var(temp)))

    index = np.array(range(len(models)))
    width = 0.3

    plt.xlabel('Context Size (n)', fontsize=16)
    plt.ylabel('Weighted Average F-score', fontsize=16)
    plt.xticks(range(len(models)), models, fontsize=12)
    count = 0
    for key in qa_label_result_means.keys():
        plt.bar(index + count*width, qa_label_result_means[key], color=colors[count], width=width, yerr=qa_label_result_sds[key])
        for a, b in zip(range(len(models)), qa_label_result_means[key]):  # 柱子上的数字显示
            plt.text(a + count*width, b, '%.1f' % b, ha='center', va='bottom', fontsize=10)
        count += 1

    # 创建图例
    plt.legend(qlabels)
    plt.show()
    #plt.savefig("./chart0612.png")

if __name__ == '__main__':
    scores = {"0": {
        "Plurality" : [86.35,84.90,85.21,86.16,85.30,86.35,83.59,86.59,86.09,85.92],
        "Definiteness": [82.35,80.55,80.04,81.20,81.61,77.37,80.80,83.59,78.69,83.05]
    },
              "1": {
                  "Plurality": [84.80,85.27,81.49,83.40,81.97,83.88,82.72,83.36,83.79,85.02],
                  "Definiteness": [81.00,79.30,79.26,79.26,77.58,76.77,78.18,80.81,78.69,79.00 ]
              },
              "2": {
                  "Plurality": [84.38,84.32,81.85,82.17,82.20,83.88,82.64,83.45,83.83,85.08],
                  "Definiteness": [81.01,80.92,79.67,80.72,78.15,75.35,77.69,80.52,77.63,80.07]
              },
              "3": {
                  "Plurality": [84.81,84.61,81.79,83.49,81.78,83.83,82.33,82.28,82.55,83.67],
                  "Definiteness": [77.61,78.60,78.10,79.93,77.75,75.05,77.37,80.00,76.37,79.57]
              },
              "4": {
                  "Plurality": [84.47,83.27,80.93,82.58,81.52,83.81,81.33,81.65,83.53,82.95],
                  "Definiteness": [79.07,79.57,79.08,80.14,78.24,76.30,78.06,79.55,76.48,79.50 ]
              }}

    drawevalplot(scores)