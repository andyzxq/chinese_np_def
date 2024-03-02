from dataset import mydata, Dataset
from model import myLSTM
from config import my_config
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn):  # 训练模型
    '''
    :param model:模型
    :param train_iterator:训练数据的迭代器
    :param dev_iterator: 验证数据的迭代器
    :param optimzer: 优化器
    :param loss_fn: 损失函数
    '''
    model.train()

    losses = []
    for i, batch in enumerate(train_iterator):
        if torch.cuda.is_available():
            input = batch.text.cuda()
            label = batch.label.type(torch.cuda.LongTensor)
        else:
            input = batch.text
            label = batch.label.type(torch.float)
            nps_list = batch.np_start
            npe_list = batch.np_end

        optimzer.zero_grad()

        pred = model(input, nps_list, npe_list)  # 预测
        pred = torch.log10(pred)    #使用nll lose, 在计算前先变成log10

        temp = []
        for slabel in label.numpy():
            if slabel[0] == 1.0:
                temp.append(0)
            else:
                temp.append(1)
        temp = torch.tensor(temp)
        #print(temp)

        loss = loss_fn.nll_loss(pred, temp)  # 计算损失值

        loss.backward()  # 误差反向传播
        losses.append(loss.data.numpy())  # 记录误差
        optimzer.step()  # 优化一次

        # 打印batch级别日志
        print(("[step = %d] loss: %.3f ") % (i + 1, loss.data.numpy() / i + 1))

        if i % 50 == 0:  # 训练50个batch后查看损失值和准确率
            avg_train_loss = np.mean(losses)
            print(f'iter:{i + 1},avg_train_loss:{avg_train_loss:.4f}')
            losses = []
            val_acc, _, _ = evaluate_model(model, dev_iterator)
            print('val_acc:{:.4f}'.format(val_acc))



def evaluate_model(model, dev_iterator):  # 评价模型
    '''
    :param model:模型
    :param dev_iterator:待评价的数据
    :return:评价（准确率）
    '''
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

def drawMarrix(y_true, y_pred, count):
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
    plt.savefig("./result/net_b512dp00h128_df_" + str(count) + ".png")

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

    optimzer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 优化器
    #loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    loss_fn = nn.functional

    #print(model)

    y = []

    for i in range(config.epoch):
        print(f'epoch:{i + 1}')
        run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn)

        # 训练一次后评估一下模型
        #train_acc, _, _ = evaluate_model(model, train_iterator)
        dev_acc, _, _ = evaluate_model(model, dev_iterator)
        test_acc, yt, yp = evaluate_model(model, test_iterator)

        print('#' * 20)
        #print('train_acc:{:.4f}'.format(train_acc))
        print('dev_acc:{:.4f}'.format(dev_acc))
        print('test_acc:{:.4f}'.format(test_acc))

        y.append(test_acc)

        # 保存模型
        torch.save(model.state_dict(), "./model/DF_B512DP00H128_" + str(i+1) + ".pt")

        #最后一次训练结束后将测试机的混淆矩阵画出来
        if i <= config.epoch - 1:
            drawMarrix(yt, yp, i+1)


    # 训练完画图
    x = [i for i in range(1,len(y) + 1)]
    fig = plt.figure()
    plt.plot(x, y, marker='o', markersize=4)
    plt.xticks(x)
    plt.savefig("./result/net_b512dp00h128_epoch_df.png")