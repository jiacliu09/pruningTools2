import os
import sys
import torch
import torchvision
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pruning_tools import pytorch_pruning

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 127, 3, 2)
        self.bn2 = torch.nn.BatchNorm2d(127)
        self.conv3 = torch.nn.Conv2d(127, 256, 3, 2)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.linear1 = torch.nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST('~/.pytorch/datasets', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST('~/.pytorch/datasets', train=False, transform=transform)

batch_size = 256
epoch = 100
step = train_data.data.shape[0] // batch_size
lr = 0.1 / 256 * batch_size

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

model = Net().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step * epoch)

##########################
prune_dict = {}
for k, v in model.named_parameters():
    if len(v.shape) != 4:
        continue
    if k == 'conv1.weight':
        prune_dict[k] = 0.9
    else:
        prune_dict[k] = 0.9
prune = pytorch_pruning.Prune(model, step * 10, step * 80, 100, prune_dict)
##########################

for idx in range(epoch):
    model.train()
    for (data, target) in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ######################
        prune.prune()
        ######################

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    # print(torch.cuda.memory_allocated(torch.device('cuda')))
    ################################################################
    layer_sparse_rate, total_sparse_rate = prune.sparsity()
    ############################################################
    print('epoch %d: Accuracy=%f; weight sparsity=%s' % (idx, test_acc, total_sparse_rate))

torch.save(model.state_dict(), 'pytorch_mnist')