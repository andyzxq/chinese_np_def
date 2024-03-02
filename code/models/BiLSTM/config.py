class my_config():
    max_length = 60  # 每句话截断长度
    word_size = 50000  # GloVe词典中词的个数
    batch_size = 256  # 一个batch的大小
    embedding_size = 100  # 词向量大小
    hidden_size = 64  # 隐藏层大小
    num_layers = 1  # 网络层数
    dropout = 0.0  # 遗忘程度
    output_size = 2  # 输出大小
    lr = 0.001  # 学习率
    epoch = 5  # 训练次数
