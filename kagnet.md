# KagNet : Knowledge-Aware Graph Networks for Commonsense Reasoning

## 1.Motivation

本文针对的数据集是：**CommonsenseQA**

目标：

- 使机器具备进行常识推理的能力，减小人机之间的差距

- - 关于推理的定义：

  - - reasoning is the process of combining facts and beliefs to make new decisions[[1\]](https://zhuanlan.zhihu.com/p/81917730#ref_1).
    - reasoning is the ability to manipulate knowledge to draw inferences [[2\]](https://zhuanlan.zhihu.com/p/81917730#ref_2).

  - 关于常识推理的定义：

  - - commonsense reasoning utilizes the basic knowledge that reflects our natural understanding of the world and human behaviors, which is common to all humans.
    - 从物理的角度：Humans' natural understanding of the physical world.
    - 从大众心理的角度：Humans' innate ability to reason about people's behavior and intentions.

- machines 缺乏可解释性：

- - how the machines manage to answer commonsense questions and make their inferences.

- 为什么要引入 **常识知识库** 以及其带来的挑战：

- - knowledge-aware models can **explicitly** incorporate external knowledge as **relational inductive biases**.

  - - **enhance reasoning capacity**;
    - increase the **transparency of model behaviors** for more interpretable results;

  - **挑战**：

  - - **noisy**: How can we ﬁnd the most relevant paths in KG?
    - **incomplete**: What if the best path is not existent in the KG?

## 1.1This work

提出了一个**Knowledge-aware reasoning 框架**，主要有以下两个步骤:

- schema graph grounding (见下图)
- graph modeling for inference

![img](https://pic4.zhimg.com/80/v2-9e5a1047afa04deaf4c285d82de87e53_720w.jpg)



提出了一个**Knowledge-aware graph network** 模块: `KAGNET`

- 核心是 `GCN-LSTM-HPA` 结构：

- - 由GCN, LSTM, 和 **hierarchical path-based attention****mechanism**组成
  - 用于 **path-based relational graph representation**

**KAGNET** 模块总体的工作流：

![img](https://pic1.zhimg.com/80/v2-56aa15b071e192aaa460512216d5fa70_720w.jpg)

- 首先，分别识别出 ![[公式]](https://www.zhihu.com/equation?tex=q) 和 ![[公式]](https://www.zhihu.com/equation?tex=a) 中提及的 concept ，根据这些 concept ，找到他们之间的路径，构建出 (ground) schema graph；
- 使用 LM encoder 编码 QA 对，产生 statement vector ![[公式]](https://www.zhihu.com/equation?tex=s) ，作为 `GCN-LSTM-HPA` 的输入，来计算 graph vector ![[公式]](https://www.zhihu.com/equation?tex=g) ；
- 最后使用 graph vector 计算最终的QA对的打分

## 2.Model

问题 ![[公式]](https://www.zhihu.com/equation?tex=q) 和一个包含 ![[公式]](https://www.zhihu.com/equation?tex=N) 选项的候选答案集 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba_i%5C%7D)

schema graph ![[公式]](https://www.zhihu.com/equation?tex=g%3D%28V%2CE%29)

### 2.1 Schema Graph Grounding

### 2.1.1 Concept Recognition

- n-gram 匹配：句子中的 token 和 ConceptNet 的顶点集合进行 n-gram 匹配
- **Note**：从有噪声的知识源中有效地提取上下文相关的知识仍是一个开放问题

### 2.1.2 Schema Graph Construction

sub-graph matching via path finding

- 采取一种直接有效的方法：直接在Q和A中提及的Concept ( ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BC%7D_q+%5Ccup+%5Cmathcal%7BC%7D_a) ) 之间查找路径

- 对于问题中的一个 Concept ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7Bc%7D_i+%5Cin+%5Cmathcal%7BC%7D_q) 和候选项中的一个 Concept ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7Bc%7D_j+%5Cin+%5Cmathcal%7BC%7D_a) ，查找他们之间路径长度小于 ![[公式]](https://www.zhihu.com/equation?tex=k+) 的path，添加到图中

- - 本文中，设置 ![[公式]](https://www.zhihu.com/equation?tex=k%3D4) ，即3-hop paths

### 2.1.3 Path Pruning via KG Embedding

为了从潜在噪声的schema graph中删除无关的path

作者使用例如TransE的知识图谱嵌入技术来预训练概念嵌入 ![[公式]](https://www.zhihu.com/equation?tex=V) 和关系类型嵌入 ![[公式]](https://www.zhihu.com/equation?tex=R) ，并且用来初始化下文提到的 ![[公式]](https://www.zhihu.com/equation?tex=KAGNET) 。作者通过将每一条路径分解成三元组来进行评分，并设置阈值删减去评分较低的路径。

- 使用KGE方法（如TransE等）预训练Concept Embedding和Relation Type Embedding（同时可用于KAGNET的初始化）
- 评价路径的质量
- 将路径分解为三元组集合，一条路径的打分为每一组三元组的乘积，通过设置一个阈值进行过滤。
- 三元组的打分通过KGE中的打分函数进行计算（例如，the confidence of triple classification）

### 2.2 Knowledge-Aware Graph Network

整体的模型结构：

![img](https://pic3.zhimg.com/80/v2-b796b496b62a15d75d6ffcf11713c46a_720w.jpg)

1. 使用GCN对图进行编码
2. 使用LSTM对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BC%7D_q) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BC%7D_q) 之间的路径进行编码，捕捉 multi-hop relational Information
3. 使用 hierarchical path-based attention 计算 relational schema graph 和 QA对 之间路径的关系

### 2.2.1 Graph Convolution Networks

使用 GCN 的目的：

1、contextually refine the concept vector

- - 这里的 context 指节点在图中的上下文，即邻接关系
  - 使用邻居来对预训练的Concept Embedding进行消歧

2、capture structural patterns of schema graphs for generalization

3、schema graph 的模式为推理提供了潜在的有价值的信息

- - QA对Concept之间的 更短、更稠密的连接 可能意味着更大的可能性，在特定的上下中。
  - 评价 候选答案 的可能性

GCN在schema graph上的计算：

- 使用预训练得到的 concept embedding 作为 GCN 计算图节点的初始化表示，即 ![[公式]](https://www.zhihu.com/equation?tex=h_i%5E%7B%280%29%7D%3DV_i)
- ![[公式]](https://www.zhihu.com/equation?tex=h_i%5E%7B%28l%2B1%29%7D+%3D+%5Csigma%28W_%7Bself%7D%5E%7B%28l%29%7D+h_i%5E%7B%28l%29%7D+%2B+%5Csum_%7Bj+%5Cin+N_i%7D+%5Cfrac%7B1%7D%7B%7CN_i%7C%7D+W%5E%7B%28l%29%7D+h_j%5E%7Bl%7D%29)

### 2.2.2 Relational Path Encoding

作者希望分别通过schema graph的角度（knowledge symbolic space）和语义信息的角度（language semantic space）获取相关知识图谱的潜在关系信息

定义问题中的第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个 concept ![[公式]](https://www.zhihu.com/equation?tex=c_i%5E%7B%28q%29%7D) 和候选答案中的第 ![[公式]](https://www.zhihu.com/equation?tex=j) 个 concept ![[公式]](https://www.zhihu.com/equation?tex=c_j%5E%7B%28a%29%7D) 之间的第 ![[公式]](https://www.zhihu.com/equation?tex=k) 条路径为 ![[公式]](https://www.zhihu.com/equation?tex=P_%7Bi%2Cj%7D%5Bk%5D) ：

- 路径是三元组序列：![[公式]](https://www.zhihu.com/equation?tex=P_%7Bi%2Cj%7D%5Bk%5D%3D%5B%28c_i%5E%7B%28q%29%7D%2C+r_0%2C+t_0%29%2C+...%2C+%28t_%7Bn-1%7D%2C+r_n%2C+c_j%5E%7B%28a%29%7D%29%5D)

- - relation vector 由 KGE 预训练得到；
  - concept vector 是 上一环节 GCN 的顶层输出；

- 每个三元组表示为: 头实体、尾实体、关系三个向量的串联，得到 triple vector；

- 使用LSTM编码三元组向量序列，得到 path vector:（使用 ![[公式]](https://www.zhihu.com/equation?tex=LSTM_%7Bs%7D) 来编码 ![[公式]](https://www.zhihu.com/equation?tex=C_%7Bq%7D) 到 ![[公式]](https://www.zhihu.com/equation?tex=C_%7Ba%7D) 之间的路径，希望捕获多跳关系信息。)

- - ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bi%2Cj%7D+%3D+%5Cfrac%7B1%7D%7B%7CP_%7Bi%2Cj%7D%7C%7D+%5Csum_k+LSTM%28P_%7Bi%2Cj%7D%5Bk%5D%29)

![[公式]](https://www.zhihu.com/equation?tex=R_%7Bi%2Cj%7D) 可以视为问题中的 concept 和 候选项中的 concept 之间的潜在的关系。

以上可以理解为schema graph中某个问题与某个答案之间所有路径的潜在关系。与之对应，

聚集所有路径的表示，得到最终的 graph vector ![[公式]](https://www.zhihu.com/equation?tex=g) ；

1、这里使用了 `Relation Network` 的方法：（可以理解为寻找陈述（问题+答案）s中的潜在语义信息。s可以从语言模型的编码器中获得（LSTM or GPT/BERT））

- - ![[公式]](https://www.zhihu.com/equation?tex=T_%7Bi%2Cj%7D+%3D+MLP%28%5Bs%3Bc_q%5E%7Bi%7D%3Bc_a%5E%7B%28j%29%7D%5D%29)
  - statement vector ![[公式]](https://www.zhihu.com/equation?tex=s) 为 LM encoder `[CLS]` 的表示
  - 关于 RN 的介绍可以参考这篇文章：

[徐阿衡：论文笔记 - A simple neural network module for relational reasoning(2017)23 赞同 · 2 评论文章![img](https://pic4.zhimg.com/v2-79ccabf40d52b713c3c5c8681bb51797_180x120.jpg)](https://zhuanlan.zhihu.com/p/34969534)

2、通过mean-pooling计算graph vector：这种计算方式称为 `GCN-LSTM-mean`     （并使用平均池化得到graph vector）

- - ![[公式]](https://www.zhihu.com/equation?tex=g%3D%5Cfrac%7B%5Csum_%7Bi%2Cj%7D%5BR_%7Bi%2Cj%7D%3BT_%7Bi%2Cj%7D%5D%7D%7B%7C%5Cmathcal%7BC%7D_q%7C+%5Ctimes+%7C%5Cmathcal%7BC%7D_a%7C%7D)

通过这种简单的方式**将分别从 `symbolic space` 和 `semantic space` 中计算的relational representation 进行融合。**

3、最终候选项的 plausibility 打分（最后进行0~1之间的评分判断问题中的某个候选答案是否可靠：）：![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D%28q%2Ca%29%3Dsigmod%28MLP%28g%29%29)



### 2.2.3 Hierarchical Attention Mechanism

考虑到不同的路径对推理的重要程度不同，采用 mean-pooling 不是一种比较可取的方式。（选择更加重要的路径和概念。）

基于此，本文提出了 **hierarchical path-based attention** 机制，有选择地聚集重要的path vector 以及更重要的QA concept 对。

分别从 path-level 和 concept-pair-level attention 来学习 根据上下文建模图表示：

1、path-level：

- ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_%7B%28i%2Cj%2Ck%29%7D+%3D+T_%7Bi%2Cj%7D+W_1+LSTM%28P_%7Bi%2Cj%7D%5Bk%5D%29)
- ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Ba%7D_%7B%28i%2Cj%2C%5Ccdot%29%7D+%3D+softmax%28%5Calpha_%7B%28i%2Cj%2Ck%29%7D%29)
- ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BR%7D_%7Bi%2Cj%7D+%3D+%5Csum_k+%5Chat%7Ba%7D_%7B%28i%2Cj%2Ck%29%7D+%5Ccdot+LSTM%28P_%7Bi%2Cj%7D%5Bk%5D%29)

2、concept-pair level：

- ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_%7B%28i%2Cj%29%7D+%3D+s+W_2+T_%7Bi%2Cj%7D)
- ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Cbeta%7D_%7B%28%5Ccdot%2C+%5Ccdot%29%7D+%3D+softmax%28%5Cbeta_%7B%28%5Ccdot%2C+%5Ccdot%29%7D%29)

3、最终的graph vector： ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bg%7D+%3D+%5Csum_%7Bi%2Cj%7D+%5Chat%7B%5Cbeta%7D_%7B%28i%2Cj%29%7D+%5B%5Chat%7BR%7D_%7B%28i%2Cj%29%7D%3B+T_%7Bi%2Cj%7D%5D)

## Experiments

### Transferability

### Case Study on Interpretibility

![img](https://pic1.zhimg.com/80/v2-6060bca542cb695d49d077a4b3fc4ab8_720w.jpg)









**从ConceptNet中构建模式子图，并通过GCN、LSTM和基于层次路径的注意力机制来学习基于路径的关系表示。**