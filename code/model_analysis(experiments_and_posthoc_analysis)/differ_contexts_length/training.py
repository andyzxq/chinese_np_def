from dataset import mydata, Dataset
from model import myBERT
from config import my_config
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].long().to(device)
        b_input_seg = batch[2].long().to(device)
        b_labels = batch[3].float()
        b_nps = batch[4].int()
        b_npe = batch[5].int()

        optimzer.zero_grad()

        pred = model(b_input_ids, b_input_mask, b_input_seg, b_nps, b_npe)  # 预测
        pred = torch.log10(pred)    #使用nll lose, 在计算前先变成log10

        temp = []
        for slabel in b_labels.to("cpu").numpy():
           for index in range(config.output_size):
               if slabel[index] == 1.0:
                    temp.append(index)
                    break
        temp = torch.tensor(temp)
        #print(temp)

        loss = loss_fn.nll_loss(pred, temp.to(device))  # 计算损失值

        loss.backward()  # 误差反向传播
        losses.append(loss.cpu().data.numpy())  # 记录误差
        optimzer.step()  # 优化一次

        # 打印batch级别日志
        print(("[step = %d] loss: %.3f ") % (i + 1, loss.cpu().data.numpy() / i + 1))

        """if i % 2000 == 0:  # 训练2000个batch后查看损失值和准确率
            avg_train_loss = np.mean(losses)
            print(f'iter:{i + 1},avg_train_loss:{avg_train_loss:.4f}')
            losses = []
            val_acc, _, _ = evaluate_model(model, dev_iterator)
            print('val_acc:{:.4f}'.format(val_acc))"""



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
        b_input_ids = batch[0].long().to(device)
        b_input_mask = batch[1].long().to(device)
        b_input_seg = batch[2].long().to(device)
        b_labels = batch[3].float()
        b_nps = batch[4].int()
        b_npe = batch[5].int()

        y_pred = model(b_input_ids, b_input_mask, b_input_seg, b_nps, b_npe)  # 预测
        predicted = torch.max(y_pred.cpu().data, 1)[1]  # 选择概率最大作为当前数据预测结果
        all_pred.extend(predicted.numpy())
        temp = []
        for slabel in b_labels.to("cpu").numpy():
            for index in range(config.output_size):
                if slabel[index] == 1.0:
                    temp.append(index)
                    break
        all_y.extend(np.array(temp))
    score = accuracy_score(all_y, np.array(all_pred).flatten())  # 计算准确率
    return score, all_y, np.array(all_pred).flatten()

if __name__ == '__main__':
    config = my_config()  # 配置对象实例化
    data_class = mydata()  # 数据类实例化
    config.output_size = data_class.n_class
    dataset = Dataset(data_class, config)  # 数据预处理实例化

    dataset.load_data()  # 进行数据预处理

    train_iterator = dataset.train_iterator  # 得到处理好的数据迭代器
    dev_iterator = dataset.dev_iterator
    test_iterator = dataset.test_iterator
    
    # 设置种子
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
      
    # 初始化模型
    model = myBERT(config)
    if torch.cuda.device_count() > 3:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # 将模型对象转变为多GPU并行运算的模型
    model = model.to(device)
    #model.load_state_dict(torch.load("./model/ClassificationV2_epoch14.pt", map_location=torch.device("cuda:3")))

    optimzer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 优化器
    #loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
    loss_fn = nn.functional

    #print(model)

    y = []

    for i in range(config.epoch):
        print(f'epoch:{i + 1}')
        run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn)

        # 训练一次后评估一下模型
        dev_acc, yt, yp = evaluate_model(model, dev_iterator)

        print('#' * 20)
        print('dev_acc:{:.4f}'.format(dev_acc))

        y.append(dev_acc)

        # 保存模型
        torch.save(model.state_dict(), "./model/bert_b48dp01_ft_df" + "_epoch" + str(i+1) + "_n4.pt")



    # 训练完画图
    x = [i for i in range(1,len(y) + 1)]
    fig = plt.figure()
    plt.plot(x, y, marker='o', markersize=4)
    plt.xticks(x)
    plt.savefig("./result/bert_b48dp01_ft_df_epoch1_5_n4.png")
