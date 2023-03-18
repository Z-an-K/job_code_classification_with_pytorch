import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from d2l import torch as d2l
import jieba




def extract_sample(file_path,is_train=True):
    data = pd.read_csv(file_path)
    df = data[["I1_4_7","I1_4_7code","I1_4_8_w16","I1_4_9_w16",
               "I1_5_7","I1_5_7code","I1_5_8_w16","I1_5_9_w16",
               "I3a_7","I3a_7code","I3a_8","I3a_9"]]
    df.columns = ["describe","code","field1","field2",
                    "describe","code","field1","field2",
                    "describe","code","field1","field2"]
    df1 = df.iloc[:,[0,1,2,3]]
    df2 = df.iloc[:,[4,5,6,7]]
    df3 = df.iloc[:,[8,9,10,11]]
    new_data = pd.concat([df1,df2,df3],ignore_index=True,join="inner")
    new_data.dropna(subset=["describe","code","field1","field2"],how="any",inplace=True,axis=0)
    new_data = new_data[new_data["describe"]!=" "]
    new_data = new_data[new_data["code"]!=" "]
    new_data = new_data[new_data["field1"]!=" "]
    new_data = new_data[new_data["field2"]!=" "]

    text = np.array(new_data).tolist()
    if is_train == True:
        text = text[0:int(len(text)*0.8)]
    else:
        text = text[int(len(text)*0.8):]
    return text

#分词
def cut_word(line):
    return jieba.lcut(line,cut_all=True)

def truncate_and_pad(line,seq_lens,pad_index):
    if len(line) > seq_lens:
        return line[:seq_lens]
    else:
        return line + [pad_index] * (seq_lens - len(line))


def map_code_to_field(code):
    code_map = {
    1:"农、林、牧、渔业",
    2:"采掘业",
    3:"制造业",
    4:"电力、煤气及水的生产和供给业",
    5:"建筑业",
    6:"质勘查业、水利管理业",
    7:"交通运输、仓储及邮电通信业",
    8:"批发和零售贸易、餐饮业",
    9:"金融保险业",
    10:"房地产业",
    11:"社会服务",
    12:"卫生、体育和社会福利业",
    13:"教育、文化艺术和广播电影电视业",
    14:"科学研究和综合技术服务业",
    15:"国家机关、党政机关和社会团体",
    99:"其他行业",
    99998:"不适用",
    99999:"不清楚"
    }
    if code not in code_map:
        code = 99999
    return code_map[code]

def get_token_and_segments(tokenA,tokenB=None):
    tokens = ["[cls]"] + tokenA + ["[sep]"]
    segments = [0] * (len(tokenA) + 2)
    if tokenB is not None:
        tokens = tokens + tokenB + ["[sep]"]
        segments = segments + [1] * (len(tokenB) + 1)
    return tokens,segments

def truncate_pair_of_tokens(tokenA,tokenB,seq_lens):
    while(len(tokenA) + len(tokenB) > seq_lens -3):
        if len(tokenA) > len(tokenB):
            tokenA.pop()
        else:
            tokenB.pop()
    return tokenA,tokenB
def pad_input(text,tokenizer,seq_lens):
    tokens,segments,field,label,attention=  [],[],[],[],[]
    label_map = {}
    i = 0
    for item in text:
        tokenA,tokenB = cut_word(item[0]),cut_word(map_code_to_field(int(item[2])))
        encoded_dict = tokenizer.encode_plus(tokenA,tokenB,max_length=seq_lens,
                                        padding="max_length",truncation=True,
                                        add_special_token=True,return_attention_mask=True)
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        token_type_ids = encoded_dict["token_type_ids"]
        attention.append(attention_masks)
        tokens.append(input_ids)
        segments.append(token_type_ids)
        #
        label.append(item[1])
        if item[1] not in label_map.keys():
            temp = {item[1]:i}
            label_map.update(temp)
            i += 1
    for i,l in enumerate(label):
        label[i] = label_map[l]

    tokens = torch.tensor(tokens)
    segments = torch.tensor(segments)
    attention = torch.tensor(attention)
    label = torch.tensor([int(l) for l in label])
    return tokens,segments,attention,label

class JobDataset(Dataset):
    def __init__(self,seq_lens,tokenizer,is_train):
        text = extract_sample("/home/zhanghan/PycharmProjects/learn_pytorch/bert-classification/data/2018.csv",is_train)
        self.seq_lens = seq_lens
        self.tokenizer = tokenizer
        self.tokens,self.segments,self.valid_lens,self.label = pad_input(text,tokenizer,seq_lens)
    def __getitem__(self, idx):
        return (self.tokens[idx],self.segments[idx],self.valid_lens[idx]),self.label[idx]

    def __len__(self):
        return len(self.tokens)



