import ast
import logging
from torch import Tensor
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm, trange
from torch.nn import BCEWithLogitsLoss
import pickle

class QAC_BERT(nn.Module):
    def __init__(self, args, vocab, char_length, num_labels=2670, load_pretrained=True):
        super().__init__()
        
        self.num_labels= num_labels
        if load_pretrained:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            config = BertConfig(vocab_size_or_config_json_file=30522)
            self.bert= BertModel(config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

class QAC_BERT_SINGLELABEL(nn.Module):
    def __init__(self, args, vocab, char_length, num_labels=2670):
        super().__init__()
        
        self.num_labels= num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_labels)
        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            _, gt_labels = torch.max(labels.data,1)
            loss = loss_fct(logits, gt_labels)
            return loss
        else:
            return logits

