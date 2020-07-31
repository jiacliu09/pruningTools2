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

**本压缩工具的优势**

    常见的压缩算法工具包只考虑压缩算法研究，训练出的模型只具有理论加速能力，在实际的硬件上运行往往无法加速或者达不到理想加速能力。与这些常见的压缩工具不同，我们的压缩工具是针对Moffett的硬件而专门产生的，可以达到理论加速能力。

    在初始化压缩器的参数中，balance这个参数就是控制是否要匹配Moffett硬件来压缩。如果设置balance=True，那么压缩出的模型在Moffett的硬件上具有理想加速能力；如果设置balance=False，那么与其他常见压缩工具一样，只具有理论加速效果，在常见硬件中无法加速。

以下是balance参数的示意图

![balance](./balance.png)



**本工程包含以下内容:**
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

[百度网盘点此下载，提取码：6ssv](https://pan.baidu.com/s/1J28WwmaYyhqSK4CWEnTLoA)

|模型|框架|训练数据集|稀疏率|准确率|说明|
|-|-|-|-|-|-|
|resnet50_v1b|mxnet|imagenet|-|77.67|gluoncv提供的预训练模型|
|resnet50_v1b|mxnet|imagenet|93.75%|74.0|基于gluoncv的预训练模型进行压缩，下同|