import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from dataprocess import Textprocess
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import statsmodels.api as sm

if __name__=="__main__":

    T = Textprocess()
    corpus, labels = T._readdata('train.csv')
    T._getMISelectKBestFeature(corpus, labels)
    words = T.vectorizer.get_feature_names()
    print(len(words))

    lr_model = LR(penalty='l1', C=2.006, solver='liblinear', max_iter=1000)

    input = open('./model/df_lr_n1.pkl', 'rb')
    lr_model = pickle.load(input)
    input.close()

    coef = lr_model.coef_  # 回归系数

    # 要把系数为零的特征从coef以及词表中删掉
    new_coef = [[]]
    vocabuary = []
    coef_dict = {}
    for i in range(len(coef[0])):
        if coef[0][i] != 0:
            coef_dict[i] = coef[0][i]

    coef_list = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)

    # top_k = 20000
    print(len(coef_list))
    for i in range(len(coef_list)):
        new_coef[0].append(coef_list[i][1])
        vocabuary.append(words[coef_list[i][0]])
        #new_coef[0].append(coef_list[-1 - i][1])
        #vocabuary.append(words[coef_list[-1 - i][0]])

    T.vectorizer = CountVectorizer(vocabulary = vocabuary)
    words = T.vectorizer.get_feature_names()
    print(len(words))
    te_corpus, te_y = T._readdata('test.csv')
    te_corpus = te_corpus[0:10]
    te_y = te_y[0:10]
    te_x = T.vectorizer.transform(te_corpus).toarray()
    """print(te_corpus[111])
    temp = te_x.toarray()[111]
    print("**********************************")
    for i in range(len(temp)):
        if temp[i] != 0:
            print(vocabuary[i])"""
    print("here")
    print(len(te_x))
    print(len(te_y))
    print(vocabuary[0:20])
    mod = sm.Logit(te_y, te_x)
    res_default = mod.fit(start_params=np.squeeze(new_coef[0]),method="bfgs", maxiter =35)
    print(res_default.summary())