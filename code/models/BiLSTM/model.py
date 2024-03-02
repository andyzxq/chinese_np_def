import torch
import torch.nn as nn
from config import my_config


class myLSTM(nn.Module):
    def __init__(self, vocab_size, config: my_config):
        super(myLSTM, self).__init__()  # 初始化
        self.vocab_size = vocab_size
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embedding_size)  # 配置嵌入层，计算出词向量
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size,  # 输入大小为转化后的词向量
            hidden_size=self.config.hidden_size,  # 隐藏层大小,
            num_layers=self.config.num_layers,  # 堆叠层数，有几层隐藏层就有几层
            dropout=self.config.dropout,  # 遗忘门参数
            bidirectional=True  # 双向LSTM
        )

        self.fc = nn.Linear(
            self.config.hidden_size * 2 * 2,  # 因为双向所有要*2,因为两个LSTM拼接所以要再*2
            self.config.output_size
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, nps_list, npe_list):
        embedded = self.embeddings(x)

        lstm_out, (_, _) = self.lstm(embedded)

        temp = []
        for i in range(lstm_out.shape[1]):
            temp.append(torch.cat([lstm_out[nps_list[i], i, :], lstm_out[npe_list[i], i, :]], dim=-1))

        # 这里将NP的第一个词和最后一个词拼接作为全连接层的输入
        fc_input = torch.stack(temp, dim=0)

        # 这里将所有隐藏层进行拼接来得出输出结果，没有使用模型的输出
        # feature_map = torch.cat([feature[i, :, :] for i in range(feature.shape[0])], dim=-1)

        out = self.fc(fc_input)

        return self.softmax(out)
