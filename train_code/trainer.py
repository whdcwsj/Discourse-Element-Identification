from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import DataLoader

from src.config import Config
from dataloader.mydataloader import BertDataset
from bert_model.original_model_bert import BertClassification
from train_code.model_train import BertTrainer
from src.config import Config



def train_bert(model, config):

    train_dataset = BertDataset(config, config.train_data_path)
    eval_dataset = BertDataset(config, config.dev_data_path)
    test_dataset = BertDataset(config, config.test_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    i = 0
    for data in train_dataloader:
        token_ids, pos, label = data
        if i == 0:
            print(token_ids.shape)  # torch.Size([1, 30, 40])
            print(token_ids)
            print(pos.shape)  # torch.Size([1, 30, 6])
            print(pos)
            print(label.shape)  # torch.Size([1, 30])
            print(label)
        i += 1




    # eval_dataloader = DataLoader(eval_dataset, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, shuffle=False)

    # trainer = BertTrainer(config, model, train_dataloader, eval_dataloader, test_dataloader)
    # trainer.train()



if __name__ == '__main__':

    config = Config(name='wsj_bert_test')
    model = BertClassification(config, bert_trainable=False).to(config.device)
    train_bert(model, config)