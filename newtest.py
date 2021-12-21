import utils
import os
import torch
import numpy as np
import newtrain
import chinese_train
import newtrain_en
import newtrain_en_ft
import argparse

np.set_printoptions(suppress=True)

role_name = ['introductionSen',
             'thesisSen',
             'ideaSen',
             'exampleSen',
             'conclusionSen',
             'otherSen',
             'evidenceSen']

summary_head = 'seed_num, best_accuracy, best_macro_f1'
csv_head = 'name, accuracy, all-p, all-r, all-f, macro-f, micro-f'
for n in role_name:
    for p in ['-p', '-r', '-f']:
        csv_head += ', ' + n + p

role_name_e = ['MajorClaim',
               'Claim',
               'Premise',
               'Other']
csv_head_e = 'name, accuracy, all-p, all-r, all-f, macro-f, micro-f'
for n in role_name_e:
    for p in ['-p', '-r', '-f']:
        csv_head_e += ', ' + n + p


def PRF(a, ignore=[]):
    precision = []
    recall = []
    f1 = []
    real = []
    TP = 0
    TPFP = 0
    TPFN = 0

    for i in range(len(a[0])):
        precision.append(a[i][i] / sum(a[:, i]))
        recall.append(a[i][i] / sum(a[i]))
        f1.append((precision[i] * recall[i] * 2) / (precision[i] + recall[i]))
        real.append(sum(a[i]))
        if i not in ignore:
            TP += a[i][i]
            TPFP += sum(a[:, i])
            TPFN += sum(a[i])

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    real = np.nan_to_num(real)
    print(precision)
    print(recall)
    print(f1)

    a_p = 0
    a_r = 0
    a_f = 0
    m_p = TP / TPFP
    m_r = TP / TPFN

    for i in range(len(f1)):
        if i not in ignore:
            a_p += real[i] * precision[i]
            a_r += real[i] * recall[i]
            a_f += real[i] * f1[i]

    total = sum(real) - sum(real[ignore])
    # print('test', total, a_p)
    print(a_p / total, a_r / total, a_f / total)

    macro_f = np.delete(f1, ignore, 0).mean()
    micro_f = (m_p * m_r * 2) / (m_p + m_r)
    print(macro_f, micro_f)
    # print(m_p, m_r)

    all_prf = [m_r, a_p / total, a_r / total, a_f / total, macro_f, micro_f]
    return precision, recall, f1, all_prf


def MeanError(a):
    n = len(a)
    MSE = 0.
    MAE = 0.
    for i in range(n):
        for j in range(n):
            if not i == j:
                MSE += (i - j) ** 2 * a[i][j]
                MAE += abs(i - j) * a[i][j]
    c = sum(sum(a))
    MSE = MSE / c
    MAE = MAE / c
    # print(sum(a))
    print(MSE, MAE)
    return MSE, MAE


def test_all(test, newdir, w_file, data, title=False, is_mask=False):
    if len(data) == 3:
        pad_documents, pad_labels, features = data
    else:
        pad_documents, pad_labels, features, essay_length = data
    with open(w_file, 'w', encoding='utf-8') as wf:
        wf.write(csv_head + '\n')
        filenames = os.listdir(newdir)

        # 保存一下最大的accurancy和macro-f1
        max_accurancy = 0
        max_macro_f1 = 0

        for file in filenames:
            fname = os.path.join(newdir, file)
            print(file)
            tag_model = torch.load(fname, map_location='cpu')
            #         tag_model.pWeight = torch.nn.Parameter(torch.ones(3))

            if len(data) == 3:
                accuracy, _, a = test(tag_model, pad_documents, pad_labels, features, 'cpu', batch_n=1, title=title,
                                   is_mask=is_mask)
            else:
                accuracy, loss, a = test(tag_model, pad_documents, pad_labels, features, essay_length, 'cpu', batch_n=1,
                                   title=title, is_mask=is_mask)
            print(accuracy)
            print(a)

            precision, recall, f1, all_prf = PRF(a[:-1, :-1], ignore=[5])
            accuracy, all_p, all_r, weighted_f, macro_f, micro_f = all_prf

            wf.write('_'.join(file.split('_')[: -1]))
            wf.write(', ' + str(accuracy))
            wf.write(', ' + str(all_p) + ', ' + str(all_r) + ', ' + str(weighted_f))
            wf.write(', ' + str(macro_f))
            wf.write(', ' + str(micro_f))

            if accuracy > max_accurancy:
                max_accurancy = accuracy
            if macro_f > max_macro_f1:
                max_macro_f1 = macro_f

            for i in range(len(f1)):
                wf.write(', ' + str(precision[i]) + ', ' + str(recall[i]) + ', ' + str(f1[i]))

            wf.write('\n')

        # 对应列保存最大的数值
        wf.write(' ')
        wf.write(', '+str(max_accurancy))
        wf.write(', ' + ', ' + ', ')
        wf.write(', ' + str(max_macro_f1))
        wf.write('\n')

    return max_accurancy, max_macro_f1


def Chinese_test(model_dir, seed, chinese_type_id):
    in_file = './data/Ch_test.json'
    embed_filename = './embd/new_embeddings2.txt'
    title = True

    print('load Chinense Embeddings...')
    max_len = 40
    embed_map, vec_size = utils.loadEmbeddings(embed_filename)

    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)

    if chinese_type_id == 0:
        pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)

    elif chinese_type_id == 3:
        pad_documents, pad_labels, essay_length = utils.sentence_padding_dgl(en_documents, en_labels, max_len, vec_size)

    is_mask = False

    if chinese_type_id == 0:
        from newtrain import test
        w_file = './newvalue/cn/' + model_package + '/%s_seed_%d.csv' % (in_file.split('.')[1].split('/')[-1], seed)

        temp_accurancy, temp_macro_f1 = test_all(test, model_dir, w_file, (pad_documents, pad_labels, features), title, is_mask=is_mask)

        return temp_accurancy, temp_macro_f1

    elif chinese_type_id == 3:
        from chinese_train import test_dgl
        w_file = './newvalue/cn/dgl/' + model_package + '/%s_seed_%d.csv' % (in_file.split('.')[1].split('/')[-1], seed)

        temp_accurancy, temp_macro_f1 = test_all(test_dgl, model_dir, w_file, (pad_documents, pad_labels, features, essay_length), title,
                                                 is_mask=is_mask)

        return temp_accurancy, temp_macro_f1


def new_Chinese_test(model_base_dir, seed, type_id):
    model_base = model_base_dir
    seed_list = seed
    i = 0

    # 存储每个seed下的accurancy和macro-f1
    accurancy_list = []
    macro_f1_list = []

    # 避免os.listdir的时候乱序读取
    path_list = os.listdir(model_base)
    path_list.sort()

    for model_file in path_list:
        print(model_base+model_file)
        temp_seed_accu, temp_seed_macro_f1 = Chinese_test(model_base+model_file, seed_list[i], type_id)
        accurancy_list.append(temp_seed_accu)
        macro_f1_list.append(temp_seed_macro_f1)
        i = i + 1

    if type_id == 0:
        summary_file = './newvalue/cn/' + model_package + '/seed_summary.csv'
    elif type_id == 3:
        summary_file = './newvalue/cn/dgl/' + model_package + '/seed_summary.csv'
    elif type_id == 4:
        summary_file = './newvalue/cn/bert/' + model_package + '/seed_summary.csv'

    j = 0
    with open(summary_file, 'w', encoding='utf-8') as wf:
        wf.write(summary_head + '\n')

        for num in seed:
            wf.write(' seed_' + str(num))
            wf.write(', ' + str(accurancy_list[j]))
            wf.write(', ' + str(macro_f1_list[j]))
            wf.write('\n')
            j = j + 1

        wf.write(' average')
        wf.write(', ' + str(np.mean(accurancy_list)))
        wf.write(', ' + str(np.mean(macro_f1_list)))
        wf.write('\n')


def test_all_e(test, newdir, w_file, data, title=False, is_mask=False, embeddings=None, ignore=[]):
    pad_documents, pad_labels, features = data

    with open(w_file, 'w', encoding='utf-8') as wf:
        wf.write(csv_head_e + '\n')
        filenames = os.listdir(newdir)

        # 保存一下最大的accurancy和macro-f1
        max_accurancy = 0
        max_macro_f1 = 0

        for file in filenames:
            fname = os.path.join(newdir, file)
            print(file)
            tag_model = torch.load(fname, map_location='cpu')

            accuracy, a = test(tag_model, pad_documents, pad_labels, features, 'cpu', batch_n=1, title=title,
                               embeddings=embeddings)

            print(accuracy)
            print(a)

            if 'e_roles_3' in newdir:
                a = a[: -1, : -1]

            precision, recall, f1, all_prf = PRF(a[1:, 1:], ignore=ignore)
            accuracy, all_p, all_r, weighted_f, macro_f, micro_f = all_prf

            wf.write('_'.join(file.split('_')[: -1]))
            wf.write(', ' + str(accuracy))
            wf.write(', ' + str(all_p) + ', ' + str(all_r) + ', ' + str(weighted_f))
            wf.write(', ' + str(macro_f))
            wf.write(', ' + str(micro_f))


            if accuracy > max_accurancy:
                max_accurancy = accuracy
            if macro_f > max_macro_f1:
                max_macro_f1 = macro_f

            for i in range(len(f1)):
                wf.write(', ' + str(precision[i]) + ', ' + str(recall[i]) + ', ' + str(f1[i]))

            wf.write('\n')

        # 对应列保存最大的数值
        wf.write(' ')
        wf.write(', '+str(max_accurancy))
        wf.write(', ' + ', ' + ', ')
        wf.write(', ' + str(max_macro_f1))
        wf.write('\n')

    return max_accurancy, max_macro_f1


def English_test(model_dir, seed):
    import utils_e as utils
    from transformers import BertTokenizer
    in_file = './data/En_test.json'
    title = True
    is_word = False
    max_len = 40

    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')

    BERT_PATH = '/home/wsj/bert_model/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    en_documents, en_labels, features = utils.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)
    pad_documents, pad_labels = utils.sentencePaddingId(en_documents, en_labels, max_len)

    from newtrain_en import test
    w_file = './newvalue/en/' + model_package + '/%s_seed_%d.csv' % (in_file.split('.')[1].split('/')[-1], seed)
    ignore = []

    temp_accurancy, temp_macro_f1 = test_all_e(test, model_dir, w_file, (pad_documents, pad_labels, features), title, embeddings=embeddings)
    return temp_accurancy, temp_macro_f1


def new_English_test(model_base_dir, seed):
    model_base = model_base_dir
    seed_list = seed
    i = 0

    # 存储每个seed下的accurancy和macro-f1
    accurancy_list = []
    macro_f1_list = []

    # 避免os.listdir的时候乱序读取
    path_list = os.listdir(model_base)
    path_list.sort()

    for model_file in path_list:
        print(model_base+model_file)
        temp_seed_accu, temp_seed_macro_f1 = English_test(model_base+model_file, seed_list[i])
        accurancy_list.append(temp_seed_accu)
        macro_f1_list.append(temp_seed_macro_f1)
        i = i + 1

    summary_file = './newvalue/en/' + model_package + '/seed_summary.csv'

    j = 0
    with open(summary_file, 'w', encoding='utf-8') as wf:
        wf.write(summary_head + '\n')

        for num in seed:
            wf.write(' seed_' + str(num))
            wf.write(', ' + str(accurancy_list[j]))
            wf.write(', ' + str(macro_f1_list[j]))
            wf.write('\n')
            j = j + 1

        wf.write(' average')
        wf.write(', ' + str(np.mean(accurancy_list)))
        wf.write(', ' + str(np.mean(macro_f1_list)))
        wf.write('\n')


def English_test_ft(model_dir, seed):
    import utils_e as utils
    from transformers import BertTokenizer
    in_file = './data/En_test.json'
    title = True
    is_word = False
    max_len = 40

    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')

    BERT_PATH = '/home/wsj/bert_model/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    en_documents, en_labels, features = utils.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)
    pad_documents, pad_labels = utils.sentencePaddingId(en_documents, en_labels, max_len)

    n_features = utils.featuresExtend(features, en_documents, en_labels, tokenizer)
    print(len(n_features[0][0]))

    from newtrain_en_ft import test
    w_file = './newvalue/enft/' + model_package + '/%s_seed_%d.csv' % (in_file.split('.')[1].split('/')[-1], seed)

    temp_accurancy, temp_macro_f1 = test_all_e(test, model_dir, w_file, (pad_documents, pad_labels, n_features), title, embeddings=embeddings)
    return temp_accurancy, temp_macro_f1

def new_English_feature_test(model_base_dir, seed):
    model_base = model_base_dir
    seed_list = seed
    i = 0

    # 存储每个seed下的accurancy和macro-f1
    accurancy_list = []
    macro_f1_list = []

    # 避免os.listdir的时候乱序读取
    path_list = os.listdir(model_base)
    path_list.sort()

    for model_file in path_list:
        print(model_base+model_file)
        temp_seed_accu, temp_seed_macro_f1 = English_test_ft(model_base+model_file, seed_list[i])
        accurancy_list.append(temp_seed_accu)
        macro_f1_list.append(temp_seed_macro_f1)
        i = i + 1

    summary_file = './newvalue/enft/' + model_package + '/seed_summary.csv'

    j = 0
    with open(summary_file, 'w', encoding='utf-8') as wf:
        wf.write(summary_head + '\n')

        for num in seed:
            wf.write(' seed_' + str(num))
            wf.write(', ' + str(accurancy_list[j]))
            wf.write(', ' + str(macro_f1_list[j]))
            wf.write('\n')
            j = j + 1

        wf.write(' average')
        wf.write(', ' + str(np.mean(accurancy_list)))
        wf.write(', ' + str(np.mean(macro_f1_list)))
        wf.write('\n')


if __name__ == "__main__":
    # 调用这个目录下的所有模型进行test
    # 0是中文，1是英文，2是英文+feature

    parser = argparse.ArgumentParser(description='Test Discourse', usage='newtest.py [<args>] [-h | --help]')
    parser.add_argument('--type_id', default=0, type=int, help='Set seed num.')
    parser.add_argument('--model_name', default='original_bert', type=str, help='set model_name')
    args = parser.parse_args()
    model_package = args.model_name

    test_type_id = args.type_id

    if test_type_id == 0:
        # model_package = newtrain.model_package_name
        model_base_dir = './newmodel/cn/' + model_package + '/'
        list_seed = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        # list_seed = [1, 100]
        new_Chinese_test(model_base_dir, list_seed, test_type_id)

    elif test_type_id == 1:
        model_package = newtrain_en.model_package_name
        model_base_dir = './newmodel/en/' + model_package + '/'
        list_seed = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        new_English_test(model_base_dir, list_seed)

    elif test_type_id == 2:
        model_package = newtrain_en_ft.model_package_name
        model_base_dir = './newmodel/enft/' + model_package + '/'
        list_seed = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        new_English_feature_test(model_base_dir, list_seed)

    elif test_type_id == 3:
        # model_package = chinese_train.model_package_name
        model_base_dir = './newmodel/cn/dgl/' + model_package + '/'
        # list_seed = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        list_seed = [1, 100]
        new_Chinese_test(model_base_dir, list_seed, test_type_id)

    elif test_type_id == 4:
        model_base_dir = './newmodel/cn/bert/' + model_package + '/'
        # list_seed = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        list_seed = [1]
        new_Chinese_test(model_base_dir, list_seed, test_type_id)


    # model_dir = './model/roles/st_rs_sppm_128_128_ap_211106182840/'
    # Chinese_test(model_dir)

    # model_dir = './model/e_roles_4/st_rs_sppm_64_64_211106193443/'
    # English_test(model_dir)

    # model_dir = './model/e_roles_3/211106211444/'
    # English_test_ft(model_dir)
