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

    lr_model = LR(penalty='l1', C=2.006, solver='liblinear', max_iter=1000)

    input = open('./model/df_lr_n15.pkl', 'rb')
    lr_model = pickle.load(input)
    input.close()

    res = lr_model.predict(te_x)

    results = []
    for i in range(len(res)):
        results.append([te_y[i], res[i]])

    results = pd.DataFrame(results)
    results.to_csv("df_lr_n15_predict.csv", header=["true", "predict"], index=False)

    print(classification_report(te_y, res, digits=4))

    # plot_confusion_matrix(lr_model, te_x, te_y)
    # plt.show()
