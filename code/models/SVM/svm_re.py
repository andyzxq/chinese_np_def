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

    svm_model = svm.SVC(C=404736.88008996064, kernel='rbf', max_iter=50000)

    input = open('./model/df_svm_n14.pkl', 'rb')
    svm_model = pickle.load(input)
    input.close()

    res = svm_model.predict(te_x)

    results = []
    for i in range(len(res)):
        results.append([te_y[i], res[i]])

    results = pd.DataFrame(results)
    results.to_csv("df_svm_n14_predict.csv", header=["true", "predict"], index=False)

    print(classification_report(te_y, res, digits=4))

    # plot_confusion_matrix(svm_model, te_x, te_y)
    # plt.show()
