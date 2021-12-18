from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import DataLoader

from src.config import Config
from dataloader.mydataloader import BertSingleDataset,BertBatchDataset
from bert_model.original_model_bert import BertClassification
from train_code.model_train import BertTrainer
from src.config import Config



def train_bert(model, config):

    train_dataset = BertBatchDataset(config=config, data_path=config.train_data_path, is_random=True)
    dev_dataset = BertSingleDataset(config, config.dev_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)

    i = 0
    for data in train_dataloader:
        token_ids, pos, label = data
        if i == 0:
            print(token_ids.shape)  # torch.Size([1, 50, 38, 40])
            print(pos.shape)  # torch.Size([1, 50, 38, 6])
            print(label.shape)  # torch.Size([1, 50, 38])
        i += 1




    # eval_dataloader = DataLoader(eval_dataset, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, shuffle=False)

    # trainer = BertTrainer(config, model, train_dataloader, eval_dataloader, test_dataloader)
    # trainer.train()



if __name__ == '__main__':

    config = Config(name='wsj_bert_test')
    model = BertClassification(config, bert_trainable=False).to(config.device)
    train_bert(model, config)