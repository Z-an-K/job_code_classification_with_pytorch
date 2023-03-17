import json
from transformers import BertTokenizer,BertConfig,BertForMaskedLM,BertForNextSentencePrediction
from transformers import BertModel
from d2l import torch as d2l
from torch import nn
import os

def load_pretrained_model(model_name,vocab_name,file_path):
    model_name = model_name
    MODEL_PATH = file_path
    tokenizer = BertTokenizer.from_pretrained(model_name)
    mode_config = BertConfig.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(MODEL_PATH,config=mode_config)
    return bert_model,tokenizer

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768,318)

    def forward(self,inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0][:,0,:]
        return self.output(self.dropout(pooled_output))

