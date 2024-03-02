import os
import pandas as pd
from sklearn import svm
from config import my_config
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import plot_confusion_matrix, classification_report
from dataprocess import Textprocess
import matplotlib.pyplot as plt
import pickle


if __name__=="__main__":
    #获得数据
    T = Textprocess()
    tr_x, tr_y, te_x, te_y = T.getData()
    #创建模型
    svm_model = svm.SVC(kernel='rbf', max_iter=1000)
    """#寻找超参数
    param_dist = {"C": sp_uniform(1E4, 1E6)}
    n_iter_search = 100
    random_search = RandomizedSearchCV(svm_model, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    sresult = random_search.fit(tr_x, tr_y)
    print("finish research hyper parameters")
    print(sresult.best_params_["C"])"""
    #设置参数开始跑
    svm_model = svm.SVC(C=191776.9819353031, kernel='rbf', max_iter=50000)
    svm_model.fit(tr_x, tr_y)
    res = svm_model.predict(te_x)
    print(classification_report(te_y, res))

    output = open('./model/ps_svm_n1.pkl', 'wb')
    s = pickle.dump(svm_model, output)
    output.close()

    """input = open('./model/ps_lr_n3.pkl', 'rb')
    lr_model = pickle.load(input)
    input.close()

    res = lr_model.predict(te_x)
    print(classification_report(te_y, res))"""

    plot_confusion_matrix(svm_model, te_x, te_y)
    plt.savefig("./svm_result/svm_n1_sp.png")
