import torch
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from torch.utils.data import DataLoader
from dataloader.mydataloader import BertSingleDataset, BertBatchDataset
from bert_model.original_model_bert import OriginalBertClassification
from train_code.model_train import BertTrainer
from src.config import Config



def train_bert(model, config):

    # is_random表示已经级取数据，不需要在DataLoader中进行shuffle
    train_dataset = BertBatchDataset(config=config, data_path=config.train_data_path, is_random=True)
    dev_dataset = BertBatchDataset(config=config, data_path=config.dev_data_path, batch_size=1, is_valid=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)

    currenttime = time.localtime()
    trainer = BertTrainer(config, model, train_data=train_dataloader, dev_data=dev_dataloader, start_time=currenttime)
    trainer.train()


if __name__ == '__main__':

    config = Config(name='original_bert')
    model = OriginalBertClassification(config, bert_trainable=False).to(config.device)
    train_bert(model, config)