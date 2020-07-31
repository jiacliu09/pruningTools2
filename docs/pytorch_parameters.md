### pytorch_pruning document：
### class Prune
### function `__init__(model, pretrain_step=0, sparse_step=0, frequency=100, prune_dict={}, restore_sparsity=False, fix_sparsity=False, balance='fix', prune_device='default')`
#### parameters

        * model

                需要压缩的模型。

        * pretrain_step (int, default=0):

                执行稀疏训练之前先执行多少步预训练。如果是从随机初始化训练模型，那么pretrain_step可以设置为适合的步数。如果是从已经训练好的模型直接压缩，那么pretrain_step可以设置为0。

        * sparse_step (int, default=0):

                执行稀疏训练多少步。在这个期间内，每frequency次就执行一次压缩。

        * frequency (int, default=100):

                执行稀疏化算法的频率。

        * prune_dict (dict, default={}):

                一个包含参数名字和目标稀疏率的字典，key为需要压缩的参数的名字，value为该参数希望达到的稀疏率。

        * restore_sparsity (bool, default=False):

                如果设置为True，则设置初始稀疏率为模型当前的稀疏率，一般用于训练中断恢复训练或finetune时使用。

        * fix_sparsity (bool, default=False):

                如果设置为True，则会固定当前模型的稀疏率不再变化，一般用于载入已经稀疏化的模型并finetune时使用。

        * balance (string, default='fix')

                与芯片相关的设置，可设置为'fix'或'none'。

        * prune_device (string, default='default')

                默认情况下稀疏操作会放在默认设备上执行，当在gpu上训练时，有时会遇到gpu显存溢出的情况，这时可设置为'cpu'，使得所有的压缩操作都放在cpu上执行，代价是训练速度会略微变慢。

### function `prune()`

        请在训练循环中，参数更新之后，调用该函数。

### function `sparsity()`

        调用该函数会返回两个dict，第一个dict包含prune_dict中所有指定需要压缩的参数的当前稀疏率，第二个dict包含prune_dict中全部参数总体的稀疏率。