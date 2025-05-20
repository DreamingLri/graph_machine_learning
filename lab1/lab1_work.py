import networkx as nx
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
说明：请大家不要改动除了`code here`之外的代码，助教会以控制台的输出结果作为评分依据
"""

# G是一个无向图
G = nx.karate_club_graph()

"""
问题1：计算`karate club network`的平均度（15分）
"""
def average_degree(num_edges, num_nodes):
    # TODO: 实现这个函数，返回值为图的平均节点度数。结果四舍五入为最近的整数
    # 提示：无向图，一边两度

    avg_degree = 0

    ############# code here ############


    ####################################

    return avg_degree
num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print("Average degree of karate club network is {}".format(avg_degree))



"""
问题 2：经过一次 PageRank 迭代后，节点 0（ID 为 0 的节点）的 PageRank 值是多少（15分）
"""
def one_iter_pagerank(G, beta, r0, node_id):
  # TODO: 实现此函数，r1值需要四舍五入到小数点后两位

  r1 = 0

  ############# code here ############
  ## Note: 不要使用 nx.pagerank 直接计算，可以通过 G.adj[node_id] 访问 node_id 的所有邻居节点
  ## r0指的是第0轮节点0的PageRank值，r1指第一轮的值； di是指节点i的度数
  ## ∑ 指的是 求和节点 i 的所有邻居节点 j


  ####################################

  return r1
beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))

def graph_to_edge_list(G):
  edge_list = list(G.edges)
  return edge_list

def edge_list_to_tensor(edge_list):
  edge_index = torch.LongTensor(edge_list).reshape(2, -1)
  return edge_index

pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)

"""
问题3：请实现以下函数。然后回答在 karate club network 中，哪些边（edge_1 到 edge_5）可能是潜在的负边？（25 分）
"""
def sample_negative_edges(G, num_neg_samples):
  # TODO: 实现返回neg_edge_list的函数。负边的采样数量为 num_neg_samples。无需考虑负边数量小于 num_neg_samples 的情况。在此实现中，自循环的边不被视为正边或负边。
  # 在这个实现中：我们对负边的定义为：图中不存在的边。
  # 其中 neg_edge_list 应该是一个数组，每个元素应该是(node_i, node_j)，其中连接node_i, node_j的边不在图中

  neg_edge_list = []

  ############# code here ############

  ####################################

  return neg_edge_list

# 采样 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# 转化为张量
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

# 那些边属于负边？
edge_1 = (7, 1)
edge_2 = (1, 33)
edge_3 = (33, 22)
edge_4 = (0, 4)
edge_5 = (4, 2)

############# code here ############
# 请打印edge1到edge5是否为负边，是负边打印True，不是打印False
def is_not_edge(edge, neg_edge_list):
  return edge in neg_edge_list or edge[::-1] in neg_edge_list


####################################

"""
问题4：为图创建节点嵌入矩阵（15分）
"""
# 不要改变这行代码
torch.manual_seed(1)
def create_node_emb(num_node=34, embedding_dim=16):
  # TODO: 返回一个 torch.nn.Embedding 层，返回层的权重矩阵应在均匀分布下初始化。

  emb = None

  ############# code here ############

  ####################################

  return emb
emb = create_node_emb()
ids = torch.LongTensor([0, 3])
print("Embedding: {}".format(emb))

def visualize_emb(emb):
  X = emb.weight.data.numpy()
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  plt.figure(figsize=(6, 6))
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  plt.scatter(club2_x, club2_y, color="blue", label="Officer")
  plt.legend()
  plt.show()


"""
问题5：训练节点嵌入（30分）
"""
# 我们希望优化节点嵌入以用于将边分类为正或负的任务。给定一个边以及每个节点的嵌入，嵌入的点积随后通过 sigmoid 函数，应该给出该边为正（sigmoid 输出大于 0.5）或负（sigmoid 输出小于 0.5）的可能性。
def accuracy(pred, label):
    # TODO: 该函数接收 pred 张量（sigmoid 之后的结果张量）和标签张量（torrent.LongTensor）。预测值(pred张量）大于 0.5 将被分类为标签 1。否则将归类为标签 0。返回的准确度应四舍五入到小数点后 4 位。
    # TODO：返回平均预测准确率
    accu = 0.0
    ############# code here ############

    ####################################
    return accu


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # TODO: 实现 train 函数， 你可以改变 训练的迭代次数以及学习率 以获得更高的性能，下面是参考的实现步骤
    # (1) 获取 train_edge 中节点的嵌入
    # (2) 对每个节点对之间的嵌入进行点积
    # (3) 将点乘结果输入 sigmoid
    # (4) 将 sigmoid 输出输入 loss_fn
    # (5) 打印每个 epoch 的损失和准确率
    # (6) 使用损失和优化器更新嵌入结果

    epochs = 500
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

    for i in range(epochs):
        ############# code here ############

        ####################################

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

train_label = torch.cat([pos_label, neg_label], dim=0)
# 由于网络非常小，我们不把边分成 val/test 集
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)

# 可视化最终结果
visualize_emb(emb)

