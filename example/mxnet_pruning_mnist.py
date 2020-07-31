import os
import sys
import mxnet
import gluoncv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pruning_tools import mxnet_pruning

class Net(mxnet.gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.conv1 = mxnet.gluon.nn.Conv2D(64, 3, 2)
        self.bn1 = mxnet.gluon.nn.BatchNorm()
        self.conv2 = mxnet.gluon.nn.Conv2D(127, 3, 2)
        self.bn2 = mxnet.gluon.nn.BatchNorm()
        self.conv3 = mxnet.gluon.nn.Conv2D(256, 3, 2)
        self.bn3 = mxnet.gluon.nn.BatchNorm()
        self.dense1 = mxnet.gluon.nn.Dense(10)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = mxnet.ndarray.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = mxnet.ndarray.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = mxnet.ndarray.relu(x)
        x = mxnet.ndarray.flatten(x)
        x = self.dense1(x)
        return x

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_gpus = 1
ctx = [mxnet.gpu(i) for i in range(num_gpus)] if num_gpus else [mxnet.cpu()]

def transform(image, label):
    image = image.astype('float32') / 255
    image = image.transpose([2, 0, 1])
    return image, label
train_data = mxnet.gluon.data.vision.MNIST('~/.mxnet/datasets/mnist', train=True, transform=transform)
test_data = mxnet.gluon.data.vision.MNIST('~/.mxnet/datasets/mnist', train=False, transform=transform)

batch_size = 256
epoch = 100
step = train_data._data.shape[0] // batch_size
lr = 0.1 / 256 * batch_size

train_loader = mxnet.gluon.data.DataLoader(train_data, batch_size, True, num_workers=8)
test_loader = mxnet.gluon.data.DataLoader(test_data, batch_size, False, num_workers=8)

model = Net()
model.initialize(mxnet.init.Xavier(magnitude=2.24), ctx=ctx)

lr_scheduler = gluoncv.utils.LRSequential([gluoncv.utils.LRScheduler('cosine', base_lr=lr, target_lr=0, nepochs=epoch, iters_per_epoch=step)])

optimizer = mxnet.optimizer.NAG(wd=1e-5, lr_scheduler=lr_scheduler)
trainer = mxnet.gluon.Trainer(model.collect_params(), optimizer)

metric = mxnet.metric.Accuracy()
softmax_cross_entropy_loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()

#################################
prune_dict = {}
for k, v in model.collect_params().items():
    if len(v.shape) != 4:
        continue
    if k == 'conv1_weight':
        prune_dict[k] = 0.9
    else:
        prune_dict[k] = 0.9
prune = mxnet_pruning.Prune(model, step * 10, step * 80, 100, prune_dict)
######################################

for i in range(epoch):
    for data, label in train_loader:
        data = mxnet.gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = mxnet.gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
        outputs = []
        with mxnet.autograd.record():
            for x, y in zip(data, label):
                z = model(x)
                loss = softmax_cross_entropy_loss(z, y)
                loss.backward()
                outputs.append(z)
        ###############
        prune.prepare()
        ###############
        trainer.step(batch_size)
        #############
        prune.prune()
        #############

    for data, label in test_loader:
        data = mxnet.gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = mxnet.gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(model(x))
        metric.update(label, outputs)
    _, test_acc = metric.get()
    metric.reset()

    # print(mxnet.context.gpu_memory_info(0)[0])
    ##################################################
    layer_sparse_rate, total_sparse_rate = prune.sparsity()
    print('epoch %d: Accuracy=%f; weight sparsity=%s' % (i, test_acc, total_sparse_rate))
    ##################################################

model.save_parameters('mxnet_mnist')