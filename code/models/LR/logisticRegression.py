import os
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
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
    lr_model = LR(penalty='l1', solver='liblinear')
    #寻找超参数
    param_dist = {"C": sp_uniform(2.0, 4.0)}
    n_iter_search = 100
    random_search = RandomizedSearchCV(lr_model, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    sresult = random_search.fit(tr_x, tr_y)
    print("finish research hyper parameters")
    print(sresult.best_params_["C"])
    #设置参数开始跑
    lr_model = LR(penalty='l1', C=sresult.best_params_["C"], solver='liblinear', max_iter=1000)
    lr_model.fit(tr_x, tr_y)
    res = lr_model.predict(te_x)
    print(classification_report(te_y, res))

    output = open('./model/ps_lr_n15.pkl', 'wb')
    s = pickle.dump(lr_model, output)
    output.close()

    """input = open('./model/ps_lr_n3.pkl', 'rb')
    lr_model = pickle.load(input)
    input.close()

    res = lr_model.predict(te_x)
    print(classification_report(te_y, res))"""

    plot_confusion_matrix(lr_model, te_x, te_y)
    plt.show()
