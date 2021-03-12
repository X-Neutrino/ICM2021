# Solution to Problem 5
import palettable
import seaborn as sns
from sklearn import datasets
from pandas import Series, DataFrame
import pandas as pd
from numpy import *
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
# Calculating the impact of node sets
# Number of simulations
max_iter_num = 1001

node = array(pd.read_csv('inode.csv', header=None))
edge = array(pd.read_csv('iedge.csv', header=None))
weight = array(pd.read_csv('iflunce.csv', header=None))
min_max = MinMaxScaler()
y = weight[:, 0]
weight[:, 1] = weight[:, 1]
w = pd.DataFrame(weight[:, 1])
w = pd.DataFrame(min_max.fit_transform(w))
w = array(w)
weight = np.column_stack((y, w))

G = nx.Graph()
[row1, col1] = node.shape
[row2, col2] = edge.shape
[row3, col3] = weight.shape
for i in range(row1):
    G.add_node(node[i][0], state=0)

for i in range(row2):
    for j in range(row3):
        if(weight[j][0] == edge[i][0]):
            G.add_edge(edge[i][0], edge[i][1], weight=weight[j][1])

seed = 'Roy Acuff'

G.nodes[seed]['state'] = 1
activated_graph = nx.Graph()  # 被激活的图
activated_graph.add_node(seed)

all_active_nodes = []  # 所有被激活的节点放在这里
all_active_nodes.append(seed)

start_influence_nodes = []  # 刚被激活的节点 即有影响力去影响别人的节点
start_influence_nodes.append(seed)


res = [[seed]]
for i in range(max_iter_num):
    new_active = list()
    t1 = '%s time' % (i*10-10) + ' %s nodes' % len(all_active_nodes)
    print(t1)  # 当前有多少个节点激活

    # 画图
    plt.title(t1)
    nx.draw(activated_graph, with_labels=False, node_size=5, width=0.1)
    plt.show()

    for v in start_influence_nodes:
        for nbr in G.neighbors(v):
            if G.nodes[nbr]['state'] == 0:  # 如果这个邻居没被激活
                edge_data = G.get_edge_data(v, nbr)
                if random.uniform(0, 1) < edge_data['weight']:
                    G.nodes[nbr]['state'] = 1
                    new_active.append(nbr)
                    activated_graph.add_edge(v, nbr)  # 画图 添加边

    print('激活', new_active)
    start_influence_nodes.clear()  # 将原先的有个影响力的清空
    start_influence_nodes.extend(new_active)  # 将新被激活的节点添加到有影响力
    all_active_nodes.extend(new_active)  # 将新被激活的节点添加到激活的列表中
    res.append(new_active)

    print('all_active_nodes', all_active_nodes)  # 打印
# print(res)

res = [c for c in res if c]
pos = nx.spring_layout(G)  # 节点的布局为spring型
nx.draw(G, pos, with_labels=False, node_color='w', node_shape='.', node_size=5, width=0.1)

for i in range(len(res)):
    nx.draw_networkx_nodes(G, pos, with_labels=False,
                           nodelist=res[i], node_size=5, width=0.1)
plt.show()
