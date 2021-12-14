import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer




class BertDataset(Dataset):
    def __init__(self, config, data_path, validation=False):
        super(BertDataset, self).__init__()
        self.config = config
        self.data_list = []
        self.tokenizer = BertTokenizer(config.vocab_path)
        if validation:
            files = os.listdir(data_path)
            for file in files:
                if '.txt' in file:
                    doc = process_doc(os.path.join(data_path, file), 'VAL')
                    self.data_list.append(doc)
        else:
            for domain in ["Business", "Politics", "Crime", "Disaster", "kbp"]:
                subdir = data_path + '/' + domain
                files = os.listdir(subdir)
                for file in files:
                    if '.txt' in file:
                        doc = process_doc(os.path.join(subdir, file), domain)
                        self.data_list.append(doc)

    def __getitem__(self, index):
        doc = self.data_list[index]
        # sent为每个段落的拆解，ls为每个段落拆解为单词后的长度，out为每个段落的labels，sids为段落编号【‘S1’,‘S2’......】
        sent, ls, out, sids = [], [], [], []
        # 首先将标题加入到sentence队列
        sent.append(' '.join(doc.headline))
        ls.append(len(doc.headline))
        # 然后将每个段落添加到列表中
        for sid in doc.sentences:
            if self.config.speech:
                out.append(self.config.out_map[doc.sent_to_speech.get(sid, 'NA')])
            else:
                out.append(self.config.out_map[doc.sent_to_event.get(sid)])
            # 将每个段落内容加入sentence队列
            sent.append(' '.join(doc.sentences[sid]))
            ls.append(len(doc.sentences[sid]))
            # 加入段落编号，目前未发现作用
            sids.append(sid)

        # tokenize
        # for sentence in sent:
        tokenized = self.tokenizer(sent, padding=True)
        token_ids = tokenized['input_ids']
        masks = tokenized['attention_mask']

        # 格式转换
        token_ids = torch.tensor(token_ids).to(self.config.device)
        masks = torch.tensor(masks).to(self.config.device)
        ls = torch.LongTensor(ls).to(self.config.device)
        out = torch.LongTensor(out).to(self.config.device)
        return token_ids, masks, ls, out, sids

    def __len__(self):
        return len(self.data_list)


class MyDataset(Dataset):
    def __init__(self, config, dataset, device, test=False):
        super(MyDataset, self).__init__()
        self.config = config
        self.dataset = dataset
        self.id_arr = np.asarray(self.dataset.iloc[:, 0])
        self.text_arr = np.asarray(self.dataset.iloc[:, 1])
        self.test = test
        if self.test is False:
            self.first_label_arr = np.asarray(self.dataset.iloc[:, 3])
            self.second_label_arr = np.asarray(self.dataset.iloc[:, 4])
        self.device = device
        if config.embedding_pretrained_model is not None:
            self.vob = config.embedding_pretrained_model.wv.key_to_index.keys()
            # # 加入PAD 字符
            # self.vob.append(0)

    def __getitem__(self, item):
        id_ = self.id_arr[item]
        id_ = torch.tensor(id_).to(self.device)
        token_ids = self.text_arr[item]
        # 处理word embedding预训练的
        if self.config.embedding_pretrained_model is not None:
            token_ids_temp = []
            # 如果不在word embedding中的token 则去掉
            for index, token_id in enumerate(token_ids):
                if token_id in self.vob:
                    # 用0号作为padding embedding多加入了一行， 所以index需要+1
                    token_ids_temp.append(self.config.embedding_pretrained_model.wv.key_to_index[token_id] + 1)
            token_ids = token_ids_temp
        # padding and truncated
        padding_len = self.config.pad_size - len(token_ids)
        if padding_len >= 0:
            token_ids = token_ids + [0] * padding_len
        else:
            token_ids = token_ids[:self.config.pad_size]
        token_ids = torch.tensor(token_ids).to(self.device)
        if self.test is False:
            first_label = self.first_label_arr[item]
            second_label = self.second_label_arr[item]
            first_label = torch.tensor(first_label - 1).to(self.device)
            second_label = torch.tensor(second_label - 1).to(self.device)
            return token_ids, first_label, second_label
        else:
            return id_, token_ids

    def __len__(self):
        return len(self.id_arr)




if __name__ == '__main__':

    config = Config(name='bert_classifier')
    model = BertClassifier(config).to(config.device)
    train_dataset = BertDataset(config, r'../data/train/train')
    dataloader = DataLoader(train_dataset)
    for data in dataloader:
        token_ids, masks, _, out, _ = data
        _output, _, _, _ = model(token_ids, masks)
        print()