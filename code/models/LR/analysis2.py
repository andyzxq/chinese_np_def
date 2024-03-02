import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from dataprocess import Textprocess
import pickle
from xpinyin import Pinyin
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import statsmodels.api as sm

if __name__=="__main__":

    T = Textprocess()
    corpus, labels = T._readdata('train.csv')
    T._getMISelectKBestFeature(corpus, labels)
    words = T.vectorizer.get_feature_names()
    print(len(words))

    # 给汉子加拼音
    newwords = []
    p = Pinyin()
    for word in words:
        wl = word.split(" ")
        temp = []
        for w in wl:
            if w != "np":
                wpy = p.get_pinyin(w, tone_marks='marks').replace("-", " ")
                temp.append(w + " (" + wpy + ")")
            else:
                temp.append("np")
        newwords.append(" ".join(temp))
    print("finish pinyin")
    feature_names = list(newwords)

    lr_model = LR(penalty='l1', C=2.006, solver='liblinear', max_iter=1000)

    input = open('./model/ps_lr_n14.pkl', 'rb')
    lr_model = pickle.load(input)
    input.close()

    coef = lr_model.coef_  # 回归系数

    # 变量重要性排序
    print(len(feature_names))
    print(len(coef.flatten()))
    coef_lr = pd.DataFrame({'var': feature_names,
                            'coef': coef.flatten()
                            })

    coef_lr = coef_lr.sort_values('coef', ascending=False)
    print(coef_lr.iloc[0:15,:])
    coef_lr_sort = coef_lr.iloc[0:15,:]
    coef_lr_sort = coef_lr_sort.append(coef_lr.iloc[len(coef_lr) - 15: len(coef_lr), :])
    print(coef_lr_sort)

    # 水平柱形图绘图
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']  # 汉字字体集
    plt.rcParams['font.size'] = 10  # 字体大小
    plt.rcParams['axes.unicode_minus'] = False
    colors = plt.cm.viridis(np.linspace(0, 1, len(coef_lr_sort)))
    fig, ax = plt.subplots()
    x, y = coef_lr_sort['var'], coef_lr_sort['coef']
    rects = plt.barh(x, y, color=colors)
    plt.grid(linestyle="-.", axis='y', alpha=0.4)
    plt.tight_layout()
    ax.set_xlabel('The value of coeffiences', fontsize=12)  # 图形的x标签
    ax.set_title('Logistic regression feature importance visualization', fontsize=16)
    # 添加数据标签
    for rect in rects:
        w = rect.get_width()
        ax.text(w, rect.get_y() + rect.get_height() / 2, '%.2f' % w, ha='left', va='center')
    plt.show()