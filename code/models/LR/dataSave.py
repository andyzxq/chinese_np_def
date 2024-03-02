import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics as mr
from sklearn import datasets
from dataprocess import Textprocess

T = Textprocess()
tr_x, tr_y, te_x, te_y = T.getData()

train_list = []
for i in range(len(tr_x)):
    train_list.append([tr_x[i], tr_y[i]])
df1 = pd.DataFrame(train_list)
df1.to_csv("train.csv", header=False, index=False)

test_list = []
for i in range(len(te_x)):
    test_list.append([te_x[i], te_y[i]])
df2 = pd.DataFrame(test_list)
df2.to_csv("test.csv", header=False, index=False)

print(te_x)
print(te_y)

"""label = [1,1,1]
word_data = ['hello yes one',
       'hi ok yes',
       'one yes']
# 实例化一个转换器类
transfer = CountVectorizer() # 词袋模型
# 调用fit_transform对原始数据进行学习
data = transfer.fit_transform(word_data) # 学习词汇表字典并返回文档术语矩阵
# 查看结果
list = []
for i in range(len(data.toarray())):
    list.append([data.toarray()[i], label[i]])
df = pd.DataFrame(list)
df.to_csv("test.csv", header=False, index=False)"""



