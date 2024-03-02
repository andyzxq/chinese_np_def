from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from config import my_config

class Textprocess:
    def __init__(self):
        self.data_dir = './data'
        self.words = []

    def _readdata(self, filename):  # 加载每行数据及其标签
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, sep=',', header=None, encoding="utf-8")
        corpus = []
        labels = []
        for index, line in df.iterrows():
            #数据由6部分组成，文本、单复数、特指泛指、NP起始位置、NP终止位置、句子词的词性
            word_list = []

            wordtag_list = line[0].split(" ")
            flag = 0
            for i in range(len(wordtag_list)):
                if "/" in wordtag_list[i]:
                    temp = wordtag_list[i].split("/")
                    if temp[1] == "PU":
                        continue
                    word_list.append(temp[0])
                elif wordtag_list[i] == "*" and flag == 0:
                    word_list.append("<NP>")
                    flag = 1
                    continue
                elif wordtag_list[i] == "*" and flag == 1:
                    word_list.append("</NP>")
                    continue

            sentence = " ".join(word_list)
            corpus.append(sentence)

            if line[1] == "S":
                ps_label = 0
            else:
                ps_label = 1

            """if line[2] == "UD":
                uddf_label = 0
            else:
                uddf_label = 1"""

            labels.append(ps_label)

        return corpus, labels

    def _getMISelectKBestFeature(self, corpus, labels):
        vectorizer = CountVectorizer(ngram_range=(1, my_config.n_gram))
        self.vectorizer = vectorizer.fit(corpus)
        print("finish word bag counter model build")

        """skb_data = vectorizer.transform(corpus)

        skb = SelectKBest(mutual_info_classif, k=my_config.features)
        self.skb = skb.fit(skb_data, labels)
        print("finish word bag MI selection model build")"""

    def _getFeatures(self, corpus):
        return self.vectorizer.transform(corpus)

    def getData(self):
        train_cropus, train_labels = self._readdata('train.csv')
        print("finish training data read")
        test_cropus, test_labels = self._readdata('test.csv')
        print("finish test data read")
        self._getMISelectKBestFeature(train_cropus, train_labels)
        print("finish features selection model build")
        train_data = self._getFeatures(train_cropus)
        print("finish training data generation")
        test_data = self._getFeatures(test_cropus)
        print("finish test data generation")

        return train_data, train_labels, test_data, test_labels
