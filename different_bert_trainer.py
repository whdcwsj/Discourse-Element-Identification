import torch
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from torch.utils.data import DataLoader
from dataloader.mydataloader import BertSingleDataset, BertBatchDataset
from bert_model.original_model_bert import OriginalBertClassification
from different_model_train_func import BertTrainer
from src.config import Config
import argparse


def train_bert(model, config, seed=None):
    # is_random表示已经随机取数据，不需要在DataLoader中进行shuffle
    train_dataset = BertBatchDataset(config=config, data_path=config.train_data_path, is_random=True)
    dev_dataset = BertBatchDataset(config=config, data_path=config.dev_data_path, batch_size=1, is_valid_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1)

    currenttime = time.localtime()
    trainer = BertTrainer(config=config, model=model, train_data=train_dataloader, dev_data=dev_dataloader,
                          start_time=currenttime, add_writer=True, seed=seed)
    trainer.train()


def test_bert(model, config, seed_list):

    test_dataset = BertBatchDataset(config=config, data_path=config.test_data_path, batch_size=1, is_valid_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    trainer = BertTrainer(config=config, model=model, test_data=test_dataloader, add_writer=False, list_seed=seed_list)

    trainer.test_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chinese Discourse',
                                     usage='different_trainer.py [<args>] [-h | --help]')
    parser.add_argument('--action', default=1, type=int, help='Train or Test.')
    parser.add_argument('--seed_num', default=1, type=int, help='Set seed num.')
    parser.add_argument('--bert_trainable', default=1, type=int, help='whether bert-chinese trainable')
    args = parser.parse_args()

    config = Config(name='bert_wwm_1_trainable')
    model = OriginalBertClassification(config, bert_trainable=args.bert_trainable).to(config.device)
    if args.action == 1:
        train_bert(model=model, config=config, seed=args.seed_num)
    elif args.action == 2:
        # 一定要在这里指定随机数种子的列表啊，！！！
        seed_list = [1, 100]
        test_bert(model=model, config=config, seed_list=seed_list)