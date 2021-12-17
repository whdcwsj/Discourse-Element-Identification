import sys
import time

import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch import optim, nn
from tqdm import tqdm


class BertTrainer:
    def __init__(self, config, model, train_data, eval_data, test_data):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.writer = SummaryWriter(log_dir=config.log_path + '/' + config.train_time)

        self.loss_function = nn.CrossEntropyLoss()

    def train(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.Adam(params, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        prev_best_perf = 0
        improve_epoch = 0
        model_save_path = self.config.model_save_path + '/model_' + self.config.train_time + '.ckpt'
        # 总loss值
        for epoch in range(self.config.epoch):
            total_loss = 0
            self.model.train()
            train_data_tqdm = tqdm(self.train_data)
            for data in train_data_tqdm:
                self.model.zero_grad()
                # optimizer.zero_grad() 和 model.zero_grad()等效
                # 获取batch信息
                token_ids, masks, _, out, _ = data
                _output, _, _, _ = self.model(token_ids, masks)
                # 计算交叉熵
                out = out.squeeze(0)
                loss = self.loss_function(_output, out)
                # 累加loss值
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                train_data_tqdm.set_description(f'Epoch {epoch}')
                train_data_tqdm.set_postfix(loss=loss.item())
                del token_ids, masks, out
                if self.config.device:
                    torch.cuda.empty_cache()

            # 使用validate_data评估当前模型
            (precision, recall, macro_f1, _), micro_f1, eval_loss = self.evaluate()
            if self.writer is not None:
                self.writer.add_scalar("loss/train_loss", total_loss / len(self.train_data), epoch)
                self.writer.add_scalar("loss/eval_loss", eval_loss / len(self.eval_data), epoch)
                self.writer.add_scalars("performance/f1", {'macro_f1': macro_f1, 'micro_f1': micro_f1}, epoch)
                self.writer.add_scalar("performance/precision", precision, epoch)
                self.writer.add_scalar("performance/recall", recall, epoch)

            if prev_best_perf < macro_f1:
                # 如果当前模型性能较好，则使用测试集进行评估，并保存模型
                prev_best_perf = macro_f1
                improve_epoch = epoch
                print("-------------------Test start-----------------------")
                self.evaluate(is_test=True)
                print("-------------------Model Saved----------------------")
                torch.save(self.model.state_dict(), model_save_path)
            elif epoch - improve_epoch >= self.config.require_improvement_epochs:
                print("model didn't improve for a long time! So break!!!")
                break
        if self.writer is not None:
            self.writer.close()

    def evaluate(self, is_test=False, need_log=False):
        y_true, y_pred = [], []
        self.model.eval()
        if is_test:
            data_tqdm = tqdm(self.test_data, desc=r"Test")
        else:
            data_tqdm = tqdm(self.eval_data, desc=r"Evaluate")
        total_loss = 0
        with torch.no_grad():
            for data in data_tqdm:
                token_ids, masks, _, out, _ = data
                out = out.squeeze(0)
                _output, _, _, _ = self.model(token_ids, masks)
                _output = _output.squeeze()
                total_loss += self.loss_function(_output, out).item()
                _, predict = torch.max(_output, 1)
                if torch.cuda.is_available():
                    predict = predict.cpu()
                    out = out.cpu()
                y_pred += list(predict.numpy())
                temp_true = list(out.numpy())
                y_true += temp_true

        macro_scores = precision_recall_fscore_support(y_true, y_pred, average='macro')
        micro_scores = precision_recall_fscore_support(y_true, y_pred, average='micro')

        print("Classification Report \n", classification_report(y_true, y_pred, digits=4))
        if is_test:
            print("MACRO: ", macro_scores)
            print("MICRO: ", micro_scores)
            print("\nConfusion Matrix \n", confusion_matrix(y_true, y_pred))

        return macro_scores, micro_scores[2], total_loss