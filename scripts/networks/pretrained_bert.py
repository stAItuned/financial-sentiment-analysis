import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

from scripts.networks.network_class import MyNetwork

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class Pretrained_Bert_Network(MyNetwork):

    def __init__(self, params):
        super().__init__(params)

        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(params['dropout'])
        self.dense = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        x = self.drop(bert_out['pooler_output'])
        out = torch.sigmoid(self.dense(x))

        return out
