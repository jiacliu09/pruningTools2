# Pruning Tools by Moffett AI
**<font size='3'>This repo provides a wide spectrum of tools for neural network pruning in major deep learning platforms, including pytorch and mxnet.**

**<font size='3'>Two modes for using the pruning tools by Moffett AI:<font>**

### 1. Prune dense networks

In order to prune their neural networks, users only need to add a few lines of codes to setup the pruning configurations and initialize the pruning process. For example:


    # use a dictionary to setup pruning configuration
        prune_dict = {}
    for k, v in model.named_parameters():
        prune_dict[k] = 0.95
    # define the
    prune = Prune(
        model = model,
        pretrian_step = 0,
        sparse_step = step * 8,
        frequency = 100,
        prune_dict = prune_dict
        balance = 'fix')

    for idx in range(epoch):
        # your training code here
        ......
        optimizer.step()
        # prune a step
        prune.prune()
        ......

    # check the sparsities of each layers during pruning
    layer_sparse_rate, total_sparse_rate = prune.sparsity()

### 2. Finetune the sparse networks on users' own datasets

Users can also use the sparse networks provided by Moffett AI to finetune on their own dataset, while the sparsity is kept.

**Notes**
Our pruning tools incorporate the option for Bank-Balanced Sparsity (BBS), which is a noval sparsity pattern that can maintain model accuracy at a high sparsity level while still enable an efficient FPGA/ASIC implementation.

The concept of BBS is shown below:
![balance](./balance.npg)



**<font size='3'>This repo includes the following contents:</font>**
1. pruning_tools文件夹包含不同框架的稀疏优化器。目前包含：
    * [x] pytorch_pruning.Prune

    * [x] mxnet_pruning.Prune

    `pytorch压缩工具文档`[请点此查看](./docs/pytorch_parameters.md)

    `mxnet压缩工具文档`[请点此查看](./docs/mxnet_parameters.md)

2. 在mnist数据集上压缩简单CNN的快速使用示例，见example文件夹：

    *从非稀疏模型训练为稀疏模型参考：*
    * [x] mxnet_pruning_mnist.py
    * [x] pytorch_pruning_mnist.py

---
我们同时提供一些已经稀疏化的预训练模型供使用，模型的稀疏率和性能见下表。模型数量会逐渐增加。目前仅提供模型，训练代码稍后也会提供。

[Baidu drive，code：6ssv](https://pan.baidu.com/s/1J28WwmaYyhqSK4CWEnTLoA)

|model|framework|training dataset|sparsity|top1|notes|
|-|-|-|-|-|-|
|resnet50_v1b|mxnet|imagenet|-|77.67|from gluoncv|
|resnet50_v1b|mxnet|imagenet|93.75%|74.0|pretrain model from gluoncv|