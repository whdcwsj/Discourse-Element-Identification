from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataloaders.dataloader import ElmoDataset, BertDataset
from src.models.GCN_model_bert import BertSAGEClassifier, GateBertSAGEClassifier
from src.models.GCN_model_elmo import ElmoGCNClassifier
from src.models.original_model_elmo import ElmoClassifier, GateElmoClassifier
from src.models.original_model_bert import BertClassifier, GateBertClassifier
from src.trainer import ElmoTrainer, BertTrainer
from src.utils.Logger import Logger



def train_bert(model, need_summary=False):
    train_dataset = BertDataset(config, r'data/train/train')
    eval_dataset = BertDataset(config, r'data/validation/validation', validation=True)
    test_dataset = BertDataset(config, r'data/test/test')
    train_dataloader = DataLoader(train_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    model.init_weights()
    trainer = BertTrainer(config, model, train_dataloader, eval_dataloader, test_dataloader, need_summary=need_summary)
    trainer.train()



if __name__ == '__main__':