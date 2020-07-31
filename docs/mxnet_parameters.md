### mxnet_pruning document：
### class Prune
```
function __init__(model, pretrain_step=0, sparse_step=0, frequency=100, prune_dict={}, restore_sparsity=False, fix_sparsity=False, balance='fix', prune_device='default')
```

        #### parameters
        * model
                model to be pruned

        * pretrain_step (int, default=0):
                pretrained steps before pruning. This value should be set to 0 if a pretrained model is loade

        * sparse_step (int, default=0):
                prune within how many steps

        * frequency (int, default=100):
                in how many steps, pruning a interval toward the target sparsity

        * prune_dict (dict, default={}):
                name-sparsity pairs with target sparsity different from the default one, such as special_sparsity_dict={'layer1.weight':0.2}

        * restore_sparsity (bool, default=False):
                restore the sparsity in the pretrained model. It is used when finetune sparse network on new dataset

        * fix_sparsity (bool, default=False):
                fix the sparsity during training. It is generally used when finetune sparse network on new datasets

        * balance (string, default='fix')
                bank balanced setup based on hardware resources. 'fix' or 'none'.

        * prune_device (string, default='default')
                on which device, the pruning operations occur. In the condition of GPU out of memory, this can be set to 'cpu', but the training speed would be slow down a little bit.

### function `prepare()`
        initialize the pruning tool before the training

### function `prune()`
        prune the weights during training after the paramters are updated

### function `sparsity()`
        monitor the sparsity during pruning