import numpy as np
import matplotlib.pyplot as plt

# 设置图形的大小
plt.figure(figsize=(10, 5))
size = 6
x = np.arange(size)
dataset = ["Introduction", "Main Idea", "Thesis", "Elaboration", "Evidence", "Conclusion"]
Bert_DCR = [0.76,
           0.58,
           0.43,
           0.55,
           0.65,
           0.83
           ]
DiSA = [0.805,
        0.55,
        0.38,
        0.66,
        0.64,
        0.84
        ]
DCRGNN = [0.81,
          0.61,
          0.44,
          0.67,
          0.695,
          0.86
          ]


# 每组实验的宽度，类别数
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2


# for i in range(1, 4):
#     plt.axhline(y=i, linestyle='--', linewidth=1.5)

# rgbcolor=[(0/255, 168/255, 225/255),
#           (224/255, 211/255, 22/255),
#           (224/255, 57/255, 11/255)]

# 画柱状图
plt.bar(x + 1 * width, Bert_DCR, width=width, label='Bert-DCR', hatch="-/-\-/", color='gainsboro', edgecolor='black')
plt.bar(x + 2 * width, DiSA, width=width, label='DiSA', hatch="**", color='gainsboro', edgecolor='black')
plt.bar(x + 3 * width, DCRGNN, width=width, label='DCRGNN', hatch="//", color='white', edgecolor='black')



#设置坐标轴范围
plt.ylim((0, 1))
# 设置坐标轴刻度
my_y_ticks = np.arange(0, 1.1, 0.1)
plt.yticks(my_y_ticks)
# 设置x轴刻度
plt.xticks(x + 2 * width, dataset)
# 设置刻度上文字的大小
plt.tick_params(labelsize=12)
# 设置坐标轴名称
# plt.xlabel('Method', fontsize=16)
plt.ylabel('Macro-F1', fontsize=12, labelpad=15)
# 设置图例位置，中上
plt.legend(loc='upper center')
# plt.show()
plt.savefig("./picture/chinese_compare2.png", format="png", dpi=1000)