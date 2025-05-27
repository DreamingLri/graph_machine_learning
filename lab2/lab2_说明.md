# **lab 2**

### 实验前准备
本次实验会使用新的数据集及部分库，其中数据集相对较大，建议同学们提前安装

#### python 库：
1. torch-scatter
2. torch-sparse
3. torch-geometric

#### 数据集：
1. Open Graph Benchmark (OGB)

### 实验目标

1. 学习使用PYG进行数据集&数据处理
2. 利用OGB数据集进行处理
3. 利用OGB数据集来建立一个简单的GCN模型并进行验证
4. 建立图神经网络GNN，并对OGB数据集进行属性预测

### 实验说明

1. 需要大家完成的任务是加粗且带有得分的题目，如 `问题 i：XXXXXXX（15分）`
2. 做完实验后，请举手通知助教检查实验代码以及问题的输出结果，以便给同学们进行打分
3. 如果大家有疑问尽量在实验课的前60分钟提出，后30分钟主要用于检查同学们的实验结果，可能时间没那么充裕

### 参考文档：
此处给出官方文档，同样推荐同学们去别的平台如stackoverflow等搜索
1. NetworkX: https://networkx.org/documentation/stable/tutorial.html
2. PyG ：https://pytorch-geometric.readthedocs.io/en/latest/
3. OGB ：https://ogb.stanford.edu/


我们将使用PyTorch Geometric（PyG）构建我们自己的图神经网络，并将模型应用于Open Graph Benchmark（OGB）数据集中的两个。 这两个数据集用于评估模型在两种不同的图相关任务上的表现。一个是节点属性预测，预测单个节点的属性。另一个是图属性预测，预测整个图或子图的属性。

注意：确保按顺序运行每个部分中的所有单元格，这样中间变量/包就可以传递到下一个单元格。

