import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    rf_model = RandomForestClassifier()
    #寻找超参数
    param_dist = {"n_estimators": [100, 200], "max_depth":[100, 500, 1000],
                  "min_samples_split":[2, 10], "min_samples_leaf":[1, 10]}
    n_iter_search = 256
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)
    sresult = random_search.fit(tr_x, tr_y)
    print("finish research hyper parameters")
    print(sresult)
    param = {"n_estimators":sresult.best_params_["n_estimators"], "max_depth":sresult.best_params_["max_depth"],
             "min_samples_split":sresult.best_params_["min_samples_split"], "min_samples_leaf":sresult.best_params_["min_samples_leaf"]}
    with open("parameters.txt", "wb", encoding="utf-8") as f:
        f.write(pickle.dumps(param))
    f.close()
    #设置参数开始跑
    rf_model = RandomForestClassifier(n_estimators=sresult.best_params_["n_estimators"], max_features="sqrt",
                                      max_depth=sresult.best_params_["max_depth"], min_samples_leaf=sresult.best_params_["min_samples_leaf"],
                                      min_samples_split=sresult.best_params_["min_samples_split"], bootstrap=True)
    """rf_model = RandomForestClassifier(n_estimators=200,
                                      max_features="sqrt",
                                      max_depth=500,
                                      min_samples_leaf=1,
                                      min_samples_split=2,
                                      bootstrap=True)"""
    rf_model.fit(tr_x, tr_y)
    res = rf_model.predict(te_x)
    print(classification_report(te_y, res))

    output = open('./model/ps_rf_n1.pkl', 'wb')
    s = pickle.dump(rf_model, output)
    output.close()

    """input = open('./model/ps_lr_n3.pkl', 'rb')
    lr_model = pickle.load(input)
    input.close()

    res = lr_model.predict(te_x)
    print(classification_report(te_y, res))"""

    plot_confusion_matrix(rf_model, te_x, te_y)
    #plt.show()
    plt.savefig("./rf_result/rf_n1_sp.png")
