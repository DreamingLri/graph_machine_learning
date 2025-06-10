"""
说明：请大家不要改动除了`code here`之外的代码，助教会以控制台的输出结果作为评分依据
"""

import networkx as nx
import matplotlib.pyplot as plt
import copy
from pylab import show

G = nx.karate_club_graph()
community_map = {}
for node in G.nodes(data=True):
    if node[1]["club"] == "Mr. Hi":
      community_map[node[0]] = 0
    else:
      community_map[node[0]] = 1
node_color = []
color_map = {0: 0, 1: 1}
node_color = [color_map[community_map[node]] for node in G.nodes()]
pos = nx.spring_layout(G)
plt.figure(figsize=(7, 7))
nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color)
show()

"""
问题1：分配 node_type 和 node_feature（30分）
"""
def assign_node_types(G, community_map):
  # TODO: 实现此函数 (10分)

  ############# code here ############

  pass
  ####################################

def assign_node_labels(G, community_map):
  # TODO: 实现此函数 (10分)

  ############# code here ############

  pass
  ####################################

def assign_node_features(G):
  # TODO: 实现此函数 (10分)

  ############# code here ############

  pass
  ####################################


assign_node_types(G, community_map)
assign_node_labels(G, community_map)
assign_node_features(G)

# Explore node properties for the node with id: 20
node_id = 20
print(f"Node {node_id} has properties:", G.nodes(data=True)[node_id])


"""
问题2：分配 edge_type（20分）
"""
def assign_edge_types(G, community_map):
  # TODO: 实现此函数

  ############# code here ############

  pass
  ####################################


assign_edge_types(G, community_map)

# Explore edge properties for a sampled edge and check the corresponding
# node types
edge_idx = 15
n1 = 0
n2 = 31
edge = list(G.edges(data=True))[edge_idx]
print(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
print(f"Node {n1} has properties:", G.nodes(data=True)[n1])
print(f"Node {n2} has properties:", G.nodes(data=True)[n2])



# 异构图可视化
edge_color = {}
for edge in G.edges():
    n1, n2 = edge
    edge_color[edge] = community_map[n1] if community_map[n1] == community_map[n2] else 2
    if community_map[n1] == community_map[n2] and community_map[n1] == 0:
      edge_color[edge] = 'blue'
    elif community_map[n1] == community_map[n2] and community_map[n1] == 1:
      edge_color[edge] = 'red'
    else:
      edge_color[edge] = 'green'

G_orig = copy.deepcopy(G)
nx.classes.function.set_edge_attributes(G, edge_color, name='color')
colors = nx.get_edge_attributes(G,'color').values()
labels = nx.get_node_attributes(G, 'node_type')
plt.figure(figsize=(8, 8))
nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color, edge_color=colors, labels=labels, font_color='white')
show()

# 转换为 `DeepSNAP` 表示
# 把 NetworkX 对象 G 转换为 deepsnap.hetero_graph.HeteroGraph
from deepsnap.hetero_graph import HeteroGraph

hete = HeteroGraph(G_orig)
"""
问题3： 每种类型的节点有多少个（15 分）
"""
def get_nodes_per_type(hete):
  # TODO: 实现此函数，给num_nodes_n0和num_nodes_n1赋值
  # 提示：使用hete.num_nodes函数
  num_nodes_n0 = 0
  num_nodes_n1 = 0

  ############# code here ############

  pass

  #####################################

  return num_nodes_n0, num_nodes_n1

num_nodes_n0, num_nodes_n1 = get_nodes_per_type(hete)
print("Node type n0 has {} nodes".format(num_nodes_n0))
print("Node type n1 has {} nodes".format(num_nodes_n1))

"""
问题4： 每种消息类型有多少条边？（15分）
"""
def get_num_message_edges(hete):
  # TODO: 实现此函数，其中message_type_edges的元素是一个元组(message_type, num_edge)
  # 提示：使用hete.num_edges函数

  message_type_edges = []

  ############# code here ############

  pass
  ####################################

  return message_type_edges


message_type_edges = get_num_message_edges(hete)
for (message_type, num_edges) in message_type_edges:
    print("Message type {} has {} edges".format(message_type, num_edges))

"""
问题5： 每个数据集拆分中有多少个节点？（20分）
"""
from deepsnap.dataset import GraphDataset

def compute_dataset_split_counts(datasets):
    # TODO: 实现此函数，返回一个字典：将数据集名称映射到该数据集中用于监督的标注的节点数量。

    data_set_splits = {}

    ############# code here ############

    ####################################

    return data_set_splits

dataset = GraphDataset([hete], task='node')
# Splitting the dataset
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}

data_set_splits = compute_dataset_split_counts(datasets)
for dataset_name, num_nodes in data_set_splits.items():
    print("{} dataset has {} nodes".format(dataset_name, num_nodes))

# `DeepSNAP` 数据集可视化
dataset = GraphDataset([hete], task='node')
# Splitting the dataset
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
titles = ['Train', 'Validation', 'Test']

for i, dataset in enumerate([dataset_train, dataset_val, dataset_test]):
    n0 = hete._convert_to_graph_index(dataset[0].node_label_index['n0'], 'n0').tolist()
    n1 = hete._convert_to_graph_index(dataset[0].node_label_index['n1'], 'n1').tolist()

    plt.figure(figsize=(7, 7))
    plt.title(titles[i])
    nx.draw(G_orig, pos=pos, node_color="grey", edge_color=colors, labels=labels, font_color='white')
    nx.draw_networkx_nodes(G_orig.subgraph(n0), pos=pos, node_color="blue")
    nx.draw_networkx_nodes(G_orig.subgraph(n1), pos=pos, node_color="red")
    show()