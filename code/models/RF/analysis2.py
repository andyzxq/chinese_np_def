from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from dataprocess import Textprocess
import pydotplus
import pickle
from xpinyin import Pinyin
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# 仍然使用自带的iris数据
#获得数据
T = Textprocess()
tr_x, tr_y, te_x, te_y = T.getData()
words = T.vectorizer.get_feature_names()

newwords = []
p = Pinyin()
# 给汉子加拼音
for word in words:
    wl = word.split(" ")
    temp = []
    for w in wl:
        if w != "np":
            wpy = p.get_pinyin(w, tone_marks='marks').replace("-", " ")
            temp.append(w +" (" + wpy + ")")
        else:
            temp.append("np")
    newwords.append(" ".join(temp))
print("finish pinyin")
feature_names = list(newwords)
target_names = ["Singular", "Plural"]

# 训练模型，限制树的最大深度4
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=500, min_samples_split=20, min_samples_leaf=1, max_features="sqrt")
input = open('./model/ps_rf_n13.pkl', 'rb')
rf_model = pickle.load(input)
input.close()

Estimators = rf_model.estimators_
for index, model in enumerate(Estimators):
    if index < 0:
        continue
    print(index)
    filename = './trees/pstree_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         max_depth = 15,
                         feature_names=feature_names,
                         class_names=target_names,
                         filled=True, rounded=True,
                         fontname="Microsoft YaHei",
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)
    if index > 9:
        break