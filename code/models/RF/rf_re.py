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
    rf_model = RandomForestClassifier(n_estimators=1000, max_depth=500, min_samples_split=20, min_samples_leaf=1, max_features="sqrt")

    input = open('./model/df_rf_n14.pkl', 'rb')
    rf_model = pickle.load(input)
    input.close()

    res = rf_model.predict(te_x)

    results = []
    for i in range(len(res)):
        results.append([te_y[i], res[i]])

    results = pd.DataFrame(results)
    results.to_csv("df_rf_n14_predict.csv", header=["true", "predict"], index=False)

    print(classification_report(te_y, res, digits=4))

    #plot_confusion_matrix(rf_model, te_x, te_y)
    #plt.show()
    # plt.savefig("./rf_result/rf_n1_sp.png")
