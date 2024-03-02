import pandas as pd
import os
import torchtext
from tqdm import tqdm


class mydata(object):
    def __init__(self):
        self.data_dir = './data'
        self.n_class = 2

    def _generator(self, filename):  # 加载每行数据及其标签
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, sep=',', header=None, encoding="utf-8")
        for index, line in df.iterrows():
            #数据由6部分组成，文本、单复数、特指泛指、NP起始位置、NP终止位置、句子词的词性
            word_list = []
            #tag_list = []
            np_start_index = -1
            np_end_index = -1

            wordtag_list = line[0].split(" ")
            flag = 0
            iindex = -1
            for i in range(len(wordtag_list)):
                if "/" in wordtag_list[i]:
                    temp = wordtag_list[i].split("/")
                    if temp[1] == "PU":
                        continue
                    word_list.append(temp[0])
                    iindex = iindex + 1
                    #tag_list.append(temp[1])
                elif wordtag_list[i] == "*" and flag == 0:
                    np_start_index = iindex + 1
                    flag = 1
                    continue
                elif wordtag_list[i] == "*" and flag == 1:
                    np_end_index = iindex
                    continue

            sentence = " ".join(word_list)
            #tag = " ".join(tag_list)

            if line[1] == "S":
                ps_label = [1.0, 0.0]
            else:
                ps_label = [0.0, 1.0]

            """if line[2] == "UD":
                uddf_label = [1, 0]
            else:
                uddf_label = [0, 1]"""

            yield sentence, ps_label, np_start_index, np_end_index

    def load_train_data(self):  # 加载数据
        return self._generator('train.csv')

    def load_dev_data(self):
        return self._generator('dev.csv')

    def load_test_data(self):
        return self._generator('test.csv')


class Dataset(object):
    def __init__(self, dataset: mydata, config):
        self.dataset = dataset
        self.config = config  # 配置文件

    def load_data(self):
        tokenizer = lambda sentence: [x for x in sentence.split() if x != ' ']  # 以空格切词
        # 定义field
        TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_length)
        LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
        NPS = torchtext.data.Field(sequential=False, use_vocab=False)
        NPE = torchtext.data.Field(sequential=False, use_vocab=False)

        # text, label能取出example对应的数据
        # 相当于定义了一种数据类型吧
        datafield = [("text", TEXT), ("label", LABEL), ("np_start", NPS), ("np_end", NPE)]

        # 加载数据
        train_gen = self.dataset.load_train_data()
        dev_gen = self.dataset.load_dev_data()
        test_gen = self.dataset.load_test_data()

        # 转换数据为example对象（数据+标签）
        train_example = [torchtext.data.Example.fromlist(it, datafield) for it in tqdm(train_gen)]
        dev_example = [torchtext.data.Example.fromlist(it, datafield) for it in tqdm(dev_gen)]
        test_example = [torchtext.data.Example.fromlist(it, datafield) for it in tqdm(test_gen)]

        # 转换成dataset
        train_data = torchtext.data.Dataset(train_example, datafield)  # example, field传入
        dev_data = torchtext.data.Dataset(dev_example, datafield)
        test_data = torchtext.data.Dataset(test_example, datafield)

        # 训练集创建字典，默认添加两个特殊字符<unk>和<pad>
        TEXT.build_vocab(train_data, max_size=self.config.word_size, vectors='glove.6B.100d')  # max_size出现频率最高的 k 个单词，加载100d的词向量
        self.vocab = TEXT.vocab        # 获取字典
        self.pretrained_embedding = TEXT.vocab.vectors  # 保存词向量

        # 放入迭代器并打包成batch及按元素个数排序，到时候直接调用即可
        self.train_iterator = torchtext.data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
        )

        self.dev_iterator, self.test_iterator = torchtext.data.BucketIterator.splits(
            (dev_data, test_data),
            batch_size=self.config.batch_size,
            sort=False,
            repeat=False,
            shuffle=False
        )

        print(f"load {len(train_data)} training examples")
        print(f"load {len(dev_data)} dev examples")
        print(f"load {len(test_data)} test examples")