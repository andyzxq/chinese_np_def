from dataset import mydata, Dataset
from model import myLSTM
from config import my_config
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def evaluate_model(model, dev_iterator):  # 评价模型
    '''
    :param model:模型
    :param dev_iterator:待评价的数据
    :return:评价（准确率）
    '''
    model.eval()
    all_pred = []
    all_y = []
    for i, batch in enumerate(dev_iterator):
        if torch.cuda.is_available():
            input = batch.text.cuda()
            label = batch.label.type(torch.cuda.LongTensor)
        else:
            input = batch.text
            label = batch.label.type(torch.float)
            nps_list = batch.np_start
            npe_list = batch.np_end

        y_pred = model(input, nps_list, npe_list)  # 预测
        predicted = torch.max(y_pred.cpu().data, 1)[1]  # 选择概率最大作为当前数据预测结果
        all_pred.extend(predicted.numpy())
        temp = []
        for slabel in label.numpy():
            if slabel[0] == 1.0:
                temp.append(0)
            else:
                temp.append(1)
        all_y.extend(np.array(temp))
    score = accuracy_score(all_y, np.array(all_pred).flatten())  # 计算准确率
    return score, all_y, np.array(all_pred).flatten()

def drawMarrix(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 可将'1'等替换成自己的类别，如'cat'。

    plt.matshow(C, cmap=plt.cm.Reds)  # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    # plt.savefig("./result/net_b512dp00h128_df_" + str(count) + ".png")
    plt.show()

if __name__ == '__main__':
    config = my_config()  # 配置对象实例化
    data_class = mydata()  # 数据类实例化
    config.output_size = data_class.n_class
    dataset = Dataset(data_class, config)  # 数据预处理实例化

    dataset.load_data()  # 进行数据预处理

    train_iterator = dataset.train_iterator  # 得到处理好的数据迭代器
    dev_iterator = dataset.dev_iterator
    test_iterator = dataset.test_iterator

    vocab_size = len(dataset.vocab)  # 字典大小

    # 初始化模型
    model = myLSTM(vocab_size, config)
    model.embeddings.weight.data.copy_(dataset.pretrained_embedding)  # 使用训练好的词向量初始化embedding层
    model.load_state_dict(torch.load("./model/SP_B256DP00H64.pt"))

    optimzer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 优化器
    #loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    loss_fn = nn.functional

    #print(model)

    test_acc, yt, yp = evaluate_model(model, test_iterator)

    results = []
    for i in range(len(yp)):
        results.append([yt[i], yp[i]])

    results = pd.DataFrame(results)
    results.to_csv("ps_BiLSTM_predict.csv", header=["true", "predict"], index=False)

    print(classification_report(yt, yp, digits=4))

    # drawMarrix(yt, yp)