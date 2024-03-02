from sklearn.metrics import classification_report
import pandas as pd

yt = []
yp = []
data = pd.read_csv("./result/df_bert_base_predict_n2.csv").values.tolist()
for item in data:
    yt.append(item[0])
    yp.append(item[1])

print(classification_report(yt, yp, digits=8))