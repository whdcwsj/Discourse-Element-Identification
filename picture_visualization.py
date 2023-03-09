# encoding: utf-8
# @author: 王宇静轩
# @file: picture_visualization.py
# @time: 2023/3/9 上午9:45


import utils
import os
import torch
import numpy as np
import newtrain
import chinese_train
import newtrain_en
import newtrain_en_ft
import argparse
import utils_e
from transformers import BertTokenizer
from sklearn.manifold import TSNE

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages(r'chinese_dataset_3d.pdf')


def Chinese_test(model_dir):
    # in_file = './data/Ch_test.json'
    in_file = './data/Ch_train.json'
    embed_filename = './embd/new_embeddings2.txt'
    title = True

    print('load Chinense Embeddings...')
    max_len = 40

    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)
    pad_documents, pad_labels, essay_length = utils.sentence_padding_dgl(en_documents, en_labels, max_len, vec_size)
    is_mask = False

    from chinese_train import picture_test_dgl

    tag_model = torch.load(model_dir, map_location='cpu')

    temp_accurancy, loss, a, sentence_list, label_list = picture_test_dgl(tag_model, pad_documents, pad_labels,
                                                                          features, essay_length, 'cpu',
                                                                          batch_n=1,
                                                                          title=title, is_mask=is_mask)

    return sentence_list, label_list


def English_test(model_dir):
    # in_file = './data/En_test.json'
    in_file = './data/En_test.json'
    title = True
    is_word = False
    max_len = 40

    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')

    BERT_PATH = './bert/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    en_documents, en_labels, features = utils_e.getEnglishSamplesBertId(in_file, tokenizer, title=title,
                                                                        is_word=is_word)
    pad_documents, pad_labels, essay_length = utils_e.sentencePaddingId_dgl(en_documents, en_labels, max_len)

    from newtrain_en import picture_test_dgl

    tag_model = torch.load(model_dir, map_location='cpu')

    temp_accurancy, loss, a, sentence_list, label_list = picture_test_dgl(tag_model, pad_documents, pad_labels,
                                                                          features, essay_length, 'cpu',
                                                                          batch_n=1,
                                                                          title=title,
                                                                          embeddings=embeddings)

    return sentence_list, label_list


if __name__ == "__main__":
    # 调用这个目录下的所有模型进行test
    # 0是中文，1是英文
    # 2是中文散点图，3是中文三维的可视化图片
    # 5是英文三维的可视化化图片

    parser = argparse.ArgumentParser(description='Test Discourse', usage='newtest.py [<args>] [-h | --help]')
    parser.add_argument('--type_id', default=5, type=int, help='Set type num.')
    args = parser.parse_args()
    test_type_id = args.type_id

    # useful_information
    # best_model_path
    # cn_best
    # /home/wsj/dgl_file/cn_best_result/SAGE部分连通/model/dgl_st_rs_sppm_128_128_ap-01-05_14.29_seed_300/dgl_st_rs_sppm_128_128_ap_t_260_best.pk
    # en_best
    # /home/wsj/dgl_file/en_best_result/model/dgl1_p4_lstm_w3_size1/dgl_st_rs_sppm_64_64_ap-03-04_16.13_seed_300/e_dgl_st_rs_sppm_64_64_ap_t_0_best.pk

    if test_type_id == 0:
        model_dir = "/home/wsj/dgl_file/cn_best_result/SAGE部分连通/model/dgl_st_rs_sppm_128_128_ap-01-05_14.29_seed_300/dgl_st_rs_sppm_128_128_ap_t_260_best.pk"
        sentences, labels = Chinese_test(model_dir)

        sentences_tune = torch.cat([i for i in sentences], 0)
        labels_tune = torch.cat([i for i in labels], 0)
        labels_tune = list(labels_tune.numpy())

        print(len(sentences_tune))
        print(len(labels_tune))

        tsne = TSNE(n_components=3, init='pca', verbose=1)
        embedding = tsne.fit_transform(sentences_tune)

        with open(r'cn_embedding.txt', 'w') as f:
            for i in embedding:
                f.write(' '.join([str(x) for x in i]))
                f.write('\n')
        # label_list += [9] * discourse_features.shape[0]
        with open(r'cn_labels.txt', 'w') as f:
            for i in labels_tune:
                f.write(str(i))
                f.write('\n')

        print(666)

    elif test_type_id == 1:
        model_dir = "/home/wsj/dgl_file/en_best_result/model/dgl1_p4_lstm_w3_size1/dgl_st_rs_sppm_64_64_ap-03-04_16.13_seed_300/e_dgl_st_rs_sppm_64_64_ap_t_0_best.pk"
        sentences, labels = English_test(model_dir)

        sentences_tune = torch.cat([i for i in sentences], 0)
        labels_tune = torch.cat([i for i in labels], 0)
        labels_tune = list(labels_tune.numpy())

        print(len(sentences_tune))
        print(len(labels_tune))

        tsne = TSNE(n_components=3, init='pca', verbose=1)
        embedding = tsne.fit_transform(sentences_tune)

        with open(r'en_embedding.txt', 'w') as f:
            for i in embedding:
                f.write(' '.join([str(x) for x in i]))
                f.write('\n')
        # label_list += [9] * discourse_features.shape[0]
        with open(r'en_labels.txt', 'w') as f:
            for i in labels_tune:
                f.write(str(i))
                f.write('\n')

        print(777)

    elif test_type_id == 2:

        embedding = []
        label_list = []
        with open(r'cn_embedding.txt') as f:
            for line in f.readlines():
                embedding.append(list(map(float, line.strip().split())))
        with open(r'cn_labels.txt') as f:
            for line in f.readlines():
                label_list.append(int(line.strip()))

        # for i in range(len(embedding) - 1, -1, -1):
        #     if label_list[i] == 0:
        #         del label_list[i]
        #         del embedding[i]
        embedding = np.array(embedding)

        out_map = {'Introduction': 0,
                   'Thesis': 1,
                   'Main Idea': 2,
                   'Evidence': 3,
                   'Conclusion': 4,
                   'Other': 5,
                   'Elaboration': 6}

        m = dict(zip(out_map.values(), out_map.keys()))

        plt.figure(figsize=(15, 11))
        # ax = plt.gca(projection='3d')
        ax = plt.subplot()
        n = 2000
        # n = embedding.shape[0]
        # 散点图
        ax.scatter(embedding[:n, 0], embedding[:n, 1],
                   s=200,
                   linewidth=2,
                   c=[(label_list[i] + 1) / 10 for i in range(n)],
                   cmap='Dark2')
        # for i in range(0, n, 2):
        #     x = embedding[i][0]
        #     y = embedding[i][1]
        #     # 设置文字说明
        #     ax.text(x, y, m[label_list[i]], fontsize=15)
        plt.show()

    elif test_type_id == 3:
        embedding = []
        label_list = []
        with open(r'cn_embedding.txt') as f:
            for line in f.readlines():
                embedding.append(list(map(float, line.strip().split())))
        with open(r'cn_labels.txt') as f:
            for line in f.readlines():
                label_list.append(int(line.strip()))

        # for i in range(len(embedding) - 1, -1, -1):
        #     if label_list[i] == 0:
        #         del label_list[i]
        #         del embedding[i]
        embedding = np.array(embedding)

        out_map = {'Introduction': 0,
                   'Thesis': 1,
                   'Main Idea': 2,
                   'Evidence': 3,
                   'Conclusion': 4,
                   'Other': 5,
                   'Elaboration': 6}
        m = dict(zip(out_map.values(), out_map.keys()))
        plt.figure(figsize=(15, 11))
        ax = plt.gca(projection='3d')
        n = 1800
        # n = embedding.shape[0]
        points = [[] for _ in range(10)]
        for i in range(n):
            points[label_list[i]].append(embedding[i])
        for i in range(10):
            points[i] = (np.mean(points[i], axis=0))
        # colors = ["icefire_r", "GnBu_r", "YlGnBu_r", "Greys", "mako_r", "rocket_r", "YlOrBr_r"]
        colors = ["mako_r", "GnBu_r", "YlGnBu_r", "Greys", "icefire_r", "rocket_r", "YlOrBr_r"]
        marks = ['.', ',', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', '+', 'D', 'd', 'x', '|', '_']
        ax.scatter(embedding[:n, 0], embedding[:n, 1], embedding[:n, 2],
                   s=60,
                   # linewidth=30,
                   c=[label_list[i] for i in range(n)],
                   marker='*',
                   # marker=[marks[label_list[i]] for i in range(n)],
                   alpha=0.8,
                   cmap='rainbow',
                   )
        for i in range(0, 7):
            ax.text(points[i][0], points[i][1], points[i][2], m[i],
                    fontsize=14,
                    verticalalignment='center',
                    horizontalalignment='center',
                    # backgroundcolor='w',
                    c='#333333',
                    alpha=0.85,
                    fontweight='semibold'
                    )
        # ax.set(xlim=[-20, 22],
        #        ylim=[-20, 14],
        #        zlim=[-18, 10],
        #        )
        # ax.view_init(30, 78)
        # 前3个参数用来调整各坐标轴的缩放比例
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.75, 1]))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # for i in range(0, n, 10):
        #     x = embedding[i][0]
        #     y = embedding[i][1]
        #     z = embedding[i][2]
        #     ax.text(x, y, z, m[label_list[i]])
        azim = -71.68831168831161
        elev = 38.713105076741385
        ax.view_init(elev, azim)
        # plt.show()
        pdf.savefig()
        plt.close()
        pdf.close()
        # print('ax.azim {}'.format(ax.azim))
        # print('ax.elev {}'.format(ax.elev))

    elif test_type_id == 5:
        embedding = []
        label_list = []
        with open(r'en_embedding.txt') as f:
            for line in f.readlines():
                embedding.append(list(map(float, line.strip().split())))
        with open(r'en_labels.txt') as f:
            for line in f.readlines():
                label_list.append(int(line.strip()))

        # for i in range(len(embedding) - 1, -1, -1):
        #     if label_list[i] == 0:
        #         del label_list[i]
        #         del embedding[i]
        embedding = np.array(embedding)

        out_map = {'padding': 0,
                   'MajorClaim': 1,
                   'Claim': 2,
                   'Premise': 3,
                   'Other': 4}
        m = dict(zip(out_map.values(), out_map.keys()))
        plt.figure(figsize=(15, 11))
        ax = plt.gca(projection='3d')
        # n = 1800
        n = embedding.shape[0]
        points = [[] for _ in range(10)]
        for i in range(n):
            points[label_list[i]].append(embedding[i])
        for i in range(10):
            points[i] = (np.mean(points[i], axis=0))
        colors = ["icefire_r", "GnBu_r", "YlGnBu_r", "Greys"]
        marks = ['.', ',', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', '+', 'D', 'd', 'x', '|', '_']
        ax.scatter(embedding[:n, 0], embedding[:n, 1], embedding[:n, 2],
                   s=60,
                   # linewidth=30,
                   c=[label_list[i] for i in range(n)],
                   marker='*',
                   # marker=[marks[label_list[i]] for i in range(n)],
                   alpha=0.8,
                   cmap='rainbow',
                   )
        # for i in range(0, 7):
        #     ax.text(points[i][0], points[i][1], points[i][2], m[i],
        #             fontsize=14,
        #             verticalalignment='center',
        #             horizontalalignment='center',
        #             # backgroundcolor='w',
        #             c='#333333',
        #             alpha=0.85,
        #             fontweight='semibold'
        #             )
        # ax.set(xlim=[-20, 22],
        #        ylim=[-20, 14],
        #        zlim=[-18, 10],
        #        )
        # ax.view_init(30, 78)
        # 前3个参数用来调整各坐标轴的缩放比例
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.75, 1]))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # for i in range(0, n, 10):
        #     x = embedding[i][0]
        #     y = embedding[i][1]
        #     z = embedding[i][2]
        #     ax.text(x, y, z, m[label_list[i]])
        # azim = -71.68831168831161
        # elev = 38.713105076741385
        # ax.view_init(elev, azim)
        plt.show()
        # pdf.savefig()
        # plt.close()
        # pdf.close()
        # print('ax.azim {}'.format(ax.azim))
        # print('ax.elev {}'.format(ax.elev))
