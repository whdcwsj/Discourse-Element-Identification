import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

pdf = PdfPages(r'dataset_3d.pdf')

embedding = []
label_list = []
with open(r'embedding_discourse.txt') as f:
    for line in f.readlines():
        embedding.append(list(map(float, line.strip().split())))
with open(r'labels_discourse.txt') as f:
    for line in f.readlines():
        label_list.append(int(line.strip()))

for i in range(len(embedding) - 1, -1, -1):
    if label_list[i] == 0:
        del label_list[i]
        del embedding[i]
embedding = np.array(embedding)

# viz_matplot(embedding)
# out_map = {'NA': 0,
#            'Main event': 1,
#            'Consequence': 2,
#            'Previous event': 3,
#            'Current Context': 4,
#            'Historical event': 5,
#            'Anecdotal event': 6,
#            'Evaluation': 7,
#            'Expectation': 8}
out_map = {'NA': 0, 'M1': 1, 'M2': 2, 'C1': 3, 'C2': 4,
           'D1': 5, 'D2': 6, 'D3': 7, 'D4': 8, 'DIS': 9}
m = dict(zip(out_map.values(), out_map.keys()))
plt.figure(figsize=(15, 11))
ax = plt.gca(projection='3d')
# n = 180
n = embedding.shape[0]
points = [[] for _ in range(10)]
for i in range(n):
    points[label_list[i]].append(embedding[i])
for i in range(10):
    points[i] = (np.mean(points[i], axis=0))
colors = ["icefire_r", "GnBu_r", "YlGnBu_r", "Greys", "mako_r", "rocket_r", "YlOrBr_r"]
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
for i in range(1, 10):
    ax.text(points[i][0], points[i][1], points[i][2], m[i],
            fontsize=15,
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
plt.show()
# pdf.savefig()
# plt.close()
# pdf.close()
