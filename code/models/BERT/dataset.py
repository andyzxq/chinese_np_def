from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import os
import pandas as pd

device_ids = [1]

class mydata(object):
    def __init__(self):
        self.data_dir = './UsedData'
        self.n_class = 2

    def _getdata(self, filename):  # 加载每行数据及其标签
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, sep=',', header=None, encoding="utf-8")
        lines = []
        labels = []
        np_s = []
        np_e = []
        for index, line in df.iterrows():
            # 数据由4部分组成，文本、标签、NP起始位置、NP终止位置
            word_list = []
            # tag_list = []
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
                    # tag_list.append(temp[1])
                elif wordtag_list[i] == "*" and flag == 0:
                    np_start_index = iindex + 1
                    flag = 1
                    continue
                elif wordtag_list[i] == "*" and flag == 1:
                    np_end_index = iindex
                    continue

            sentence = "".join(word_list)
            # tag = " ".join(tag_list)

            """if line[1] == "S":
                ps_label = [1.0, 0.0]
            else:
                ps_label = [0.0, 1.0]"""

            if line[2] == "UD":
                uddf_label = [1, 0]
            else:
                uddf_label = [0, 1]

            #计算wordpieces位置
            np_ts = 0
            np_te = 0
            for i in range(len(word_list)):
                if i < np_start_index:
                    np_ts += len(word_list[i])
                    np_te += len(word_list[i])
                elif i <= np_end_index and i >= np_start_index:
                    np_te += len(word_list[i])
            np_te -= 1

            lines.append(sentence)
            labels.append(uddf_label)
            np_s.append(np_ts)
            np_e.append(np_te)

        return lines, labels, np_s, np_e

    def load_train_data(self):  # 加载数据
        return self._getdata('train.csv')

    def load_dev_data(self):
        return self._getdata('dev.csv')

    def load_test_data(self):
        return self._getdata('test.csv')


class Dataset(object):
    def __init__(self, dataset: mydata, config):
        self.dataset = dataset
        self.config = config  # 配置文件

    # 将每一句转成数字 （大于510做截断，小于510做 Padding，加上首位两个标识，长度总共等于128）
    def convert_text_to_token(self, tokenizer, sentence):
        limit_length = self.config.max_length
        # 直接截断
        if len(sentence) > limit_length:
            sentence = sentence[:limit_length]

        tokens = tokenizer.encode(sentence)

        if len(tokens) < limit_length + 2:  # 补齐（pad的索引号就是0）
            tokens.extend([0] * (limit_length + 2 - len(tokens)))

        #print(tokens)

        seg = [0 for i in range(len(tokens))]

        return tokens, seg

    # 建立mask
    def attention_masks(self, input_ids):
        atten_masks = []
        for seq in input_ids:  # [10000, 512]
            seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
            atten_masks.append(seq_mask)
        return atten_masks

    def load_data(self):
        # 加载数据
        train_lines, train_labels, train_nps, train_npe = self.dataset.load_train_data()
        dev_lines,  dev_labels, dev_nps, dev_npe = self.dataset.load_dev_data()
        test_lines, test_labels, test_nps, test_npe = self.dataset.load_test_data()

        print("finish data load")

        tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')

        input_train_ids = []
        seg_train_ids = []

        for i in range(len(train_lines)):
            ids, seg = self.convert_text_to_token(tokenizer, train_lines[i])
            input_train_ids.append(ids)
            seg_train_ids.append(seg)

        input_dev_ids = []
        seg_dev_ids = []

        for i in range(len(dev_lines)):
            ids, seg = self.convert_text_to_token(tokenizer, dev_lines[i])
            input_dev_ids.append(ids)
            seg_dev_ids.append(seg)

        input_test_ids = []
        seg_test_ids = []

        for i in range(len(test_lines)):
            ids, seg = self.convert_text_to_token(tokenizer, test_lines[i])
            input_test_ids.append(ids)
            seg_test_ids.append(seg)


        input_train_seg = torch.tensor(seg_train_ids, device='cuda:'+str(device_ids[0]))
        input_dev_seg = torch.tensor(seg_dev_ids, device='cuda:'+str(device_ids[0]))
        input_test_seg = torch.tensor(seg_test_ids, device='cuda:'+str(device_ids[0]))

        input_train_tokens = torch.tensor(input_train_ids, device='cuda:'+str(device_ids[0]))
        input_dev_tokens = torch.tensor(input_dev_ids, device='cuda:'+str(device_ids[0]))
        input_test_tokens = torch.tensor(input_test_ids, device='cuda:'+str(device_ids[0]))

        atten_train_masks = self.attention_masks(input_train_ids)
        atten_dev_masks = self.attention_masks(input_dev_ids)
        atten_test_masks = self.attention_masks(input_test_ids)

        attention_train_tokens = torch.tensor(atten_train_masks, device='cuda:'+str(device_ids[0]))
        attention_dev_tokens = torch.tensor(atten_dev_masks, device='cuda:'+str(device_ids[0]))
        attention_test_tokens = torch.tensor(atten_test_masks, device='cuda:'+str(device_ids[0]))

        train_labels = torch.tensor(train_labels, device='cuda:'+str(device_ids[0]))
        dev_labels = torch.tensor(dev_labels, device='cuda:'+str(device_ids[0]))
        test_labels = torch.tensor(test_labels, device='cuda:'+str(device_ids[0]))

        train_nps = torch.tensor(train_nps, device='cuda:'+str(device_ids[0]))
        dev_nps = torch.tensor(dev_nps, device='cuda:' + str(device_ids[0]))
        test_nps = torch.tensor(test_nps, device='cuda:' + str(device_ids[0]))

        train_npe = torch.tensor(train_npe, device='cuda:' + str(device_ids[0]))
        dev_npe = torch.tensor(dev_npe, device='cuda:' + str(device_ids[0]))
        test_npe = torch.tensor(test_npe, device='cuda:' + str(device_ids[0]))

        print("完成数据的BERT预处理")

        train_data = TensorDataset(input_train_tokens, attention_train_tokens, input_train_seg, train_labels, train_nps, train_npe)
        train_sampler = RandomSampler(train_data)
        self.train_iterator = DataLoader(train_data, sampler=train_sampler, batch_size=self.config.batch_size)

        dev_data = TensorDataset(input_dev_tokens, attention_dev_tokens, input_dev_seg, dev_labels, dev_nps, dev_npe)
        dev_sampler = RandomSampler(dev_data)
        self.dev_iterator = DataLoader(dev_data, sampler=dev_sampler, batch_size=self.config.batch_size)

        test_data = TensorDataset(input_test_tokens, attention_test_tokens, input_test_seg, test_labels, test_nps, test_npe)
        test_sampler = RandomSampler(test_data)
        self.test_iterator = DataLoader(test_data, sampler=test_sampler, batch_size=self.config.batch_size)

        print(f"load {len(train_data)} training examples")
        print(f"load {len(dev_data)} dev examples")
        print(f"load {len(test_data)} test examples")
