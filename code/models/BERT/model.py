import torch
import torch.nn as nn
from config import my_config
from transformers import BertConfig, BertModel


class myBERT(nn.Module):
    def __init__(self, config: my_config):
        super(myBERT, self).__init__()  # 初始化
        self.config = config
        self.bert_config = BertConfig.from_pretrained('hfl/chinese-bert-wwm')
        self.bert_module = BertModel.from_pretrained('hfl/chinese-bert-wwm', config=self.bert_config)
        self.dropout_layer = nn.Dropout(self.config.dropout)
        out_dims = self.bert_config.hidden_size
        self.fc = nn.Linear(2*out_dims, self.config.output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, input_mask, segment_ids, nps_list, npe_list):
        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )

        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]
        # 对反向传播及逆行截断,也就是不更新BERT参数
        # x = seq_out.detach()
        x = seq_out

        # 将np头尾的token的词向量拼接在一起
        temp = []
        for i in range(seq_out.shape[0]):
            temp.append(torch.cat([seq_out[i, nps_list[i], :], seq_out[i, npe_list[i], :]], dim=-1))

        fc_input = torch.stack(temp, dim=0)

        out = self.fc(fc_input)

        return self.softmax(out)
