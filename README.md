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
![balance](./balance.png)


### This repo includes the following contents:

#### 1. pruning tools used in pytorch and mxnet：

    [x] pytorch_pruning.Prune

    [x] mxnet_pruning.Prune

Detailed documents for pruning optimizers:

[doc for pytorch pruning tools](./docs/pytorch_parameters.md)

[doc for mxnet pruning tools](./docs/mxnet_parameters.md)

2. Examples of using pruning optimizers on mnist dataset:
    [x] mxnet_pruning_mnist.py
    [x] pytorch_pruning_mnist.py

---
**<font size='3'>4. Moffett AI model zoo </font>**

We provide some sparse networks for users to finetune on their own datasets. More sparse networks will be constantly provided in this repo.

[Baidu drive，code：6ssv](https://pan.baidu.com/s/1J28WwmaYyhqSK4CWEnTLoA)

|model|framework|training dataset|sparsity|top1|notes|
|-|-|-|-|-|-|
|resnet50_v1b|mxnet|imagenet|0|77.67|from gluoncv|
|resnet50_v1b|mxnet|imagenet|93.75%|74.0|pretrain model from gluoncv|