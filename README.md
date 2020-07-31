# Pruning Tools
## 本工程是`Moffett AI`开发的神经网络剪枝工具
使用剪枝优化器在深度神经网络模型训练过程对其进行剪枝，达到大幅降低运算量从而加速推理的目的。

**剪枝工具使用很方便:**
* 用户可将已有训练代码的optimizer直接替换为本工程的提供的优化器，从而直接压缩模型，而几乎不需要修改其他代码。例如：
    ```key
    # 在训练循环之前，需要创建一个dict，dict的key是需要压缩的参数的名称，value是该参数对应的需要压缩的比例，然后初始化压缩器。
    prune_dict = {}
    for k, v in model.named_parameters():
        prune_dict[k] = 0.95
    prune = Prune(model, step * 0, step * 8, 10, prune_dict)

    for idx in range(epoch):
        # your training code here
        ......
        optimizer.step()
        # 在执行更新梯度操作后，调用以下函数
        prune.prune()
        ......

    # 如果希望查看dict中指定的参数当前的压缩率是多少，可以调用这个函数，它会输出每个参数当前的压缩率，已经整体的压缩率。
    layer_sparse_rate, total_sparse_rate = prune.sparsity()
    ```

* 也可将模型替换成我们提供的稀疏化预训练模型，在用户数据上保持稀疏率的前提下进行finetune。

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