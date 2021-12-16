import json
import os, sys
import numpy as np
import math

import random

# random.seed(312)

# 总训练集长度
# 1112

def splitDataIntoTwoPart(in_file, out_file1, out_file2, data_length=1112, p=0.1):
    # 使用random.seed()，而不是np.numpy.seed()，抱着测试集的不变性，避免主项目中的随机数变化干扰数据划分
    random.seed(312)
    # 返回范围[a,b]内的随机整数
    # 取原训练集10%的数字作为验证集
    # random.randint()会存在重复(取值范围[a,b])
    # dev_idx = [random.randint(0, data_length - 1) for _ in range(int(data_length * p))]
    dev_idx = random.sample(range(0, data_length), int(data_length * p))
    # print(len(dev_idx))
    # print(dev_idx)

    train_file = open(out_file1, 'w', encoding='utf-8')
    dev_file = open(out_file2, 'w', encoding='utf-8')

    count = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        for i in range(data_length):
            temp_data = f.readline()
            if count in dev_idx:
                dev_file.write(temp_data)
            else:
                train_file.write(temp_data)
            count += 1

    train_file.close()
    dev_file.close()


# 用不上了
# def modifyCombinedSentence(in_file, out_file, data_length=111):
#
#
#     combine_file = open(out_file, 'w', encoding='utf-8')
#
#     count = 0
#     with open(in_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             if count == 0:
#
#                 # 这样读的数据都是str格式的，不太好处理
#                 # temp_data = f.readline()
#                 # print(temp_data)
#                 # print(type(temp_data))
#
#                 temp_data = json.loads(line)
#                 word_sentence = temp_data['sents']
#
#                 print(word_sentence)
#                 print(len(word_sentence))
#
#                 for j in range(len(word_sentence)):
#
#             count += 1
#
#
#     combine_file.close()


if __name__ == "__main__":
    # dev_id_show = [297, 782, 66, 303, 669, 474, 394, 934, 903, 247, 609, 681, 314, 11, 653, 554, 334, 309, 517, 97, 660,
    #                563, 651, 417, 223, 370, 794, 441, 487, 62, 252, 943, 282, 632, 382, 910, 665, 388, 395, 48, 341,
    #                156, 352, 399, 935, 359, 405, 498, 448, 992, 201, 440, 1057, 627, 222, 1036, 732, 859, 850, 2, 738,
    #                244, 155, 213, 608, 83, 682, 484, 469, 997, 590, 377, 484, 191, 384, 649, 403, 414, 395, 322, 230,
    #                856, 963, 278, 941, 901, 1026, 87, 388, 532, 555, 443, 892, 571, 1050, 1078, 1084, 1062, 900, 1084,
    #                1014, 344, 80, 587,489, 722, 1047, 430, 452, 267, 685]
    #
    # print(len(dev_id_show))
    # 111

    file_in = './Ch_train.json'
    file_out1 = './new_Ch_train.json'
    file_out2 = './new_Ch_dev.json'
    splitDataIntoTwoPart(in_file=file_in, out_file1=file_out1, out_file2=file_out2)

    # file_in = './new_Ch_dev.json'
    # file_out = './new_Combine_Ch_dev.json'
    # modifyCombinedSentence(in_file=file_in, out_file=file_out)
