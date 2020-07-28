import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np

# import PIL.ImageDraw as Image
# import cv2 as cv

##############By 2rintf###############
#! LeNet-5 on FashionMNIST.          #
#! Optimizer: Adam.                  #
#! Loss func: Cross-Entropy.         #
#! Result: Train_accuracy = 95.113%  #
#           Test_accuracy = 90.04%   #
######################################



# param
EPOCH = 20
BATCH_SIZE = 32
LR =0.001
LR2 = 0.0003
DOWNLOAD=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,0),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.out = nn.Sequential(
            nn.Linear(4*4*16,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)         # -> (batch, 16, 5, 5)
        x = x.view(x.size(0),-1)  # flatern 展平 (batch,5*5*16)
        output = self.out(x)
        return output

net = CNN()
print(net)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(),LR)
loss_func = nn.CrossEntropyLoss()


# data preprocess
classes = (0,1,2,3,4,5,6,7,8,9)
train_data= torchvision.datasets.FashionMNIST(
    root="./MNIST/FashionMNIST",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD
)
print(train_data.data.shape)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data= torchvision.datasets.FashionMNIST(
    root="./MNIST/FashionMNIST",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD
)
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)
print(test_data.data.shape)

# train
net.train()
for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    if epoch > 9:
        optimizer = torch.optim.Adam(net.parameters(),LR2)

    if epoch%1==0:
        # train_data test
        correct_t = 0
        total_t = 0
        for data in train_loader:
            images_t, labels_t = data
            outputs_t = net(images_t.cuda())
            _, predicted_t = torch.max(outputs_t.data, 1)
            total_t += labels_t.size(0)
            correct_t += (predicted_t == labels_t.cuda()).sum().item()  # item() -> Tensor转标量

        print('epoch: %d | Train dataset accuracy: %.3f %%' % (epoch, 100 * correct_t*1.0 / total_t))

        # test_data test
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            outputs = net(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()  # item() -> Tensor转标量

        print('epoch: %d | Test accuracy: %.3f %%' % (epoch, 100 * correct * 1.0/ total))

    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        output = net(inputs.cuda())
        loss = loss_func(output, labels.cuda())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
print('Finished Training')

# torch.save(net.state_dict(),"./FM_lenet_20E_32B_2LR.pth")

# final result
correct_t = 0
total_t = 0
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images_t, labels_t = data
        outputs_t = net(images_t.cuda())
        _, predicted_t = torch.max(outputs_t.data, 1)
        total_t += labels_t.size(0)
        correct_t += (predicted_t == labels_t.cuda()).sum().item()  # item() -> Tensor转标量

    print('FINAL | Train dataset accuracy: %.3f %%' % ( 100 * correct_t*1.0 / total_t))


    for data in test_loader:
        images, labels = data
        outputs = net(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item() # item() -> Tensor转标量
    print('FINAL | Test dataset accuracy: %.3f %%' % (100 * correct / total))