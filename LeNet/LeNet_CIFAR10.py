import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np

# import PIL.ImageDraw as Image
# import cv2 as cv
import matplotlib.pyplot as plt


##############By 2rintf###############
#! LeNet-5 on CIFAR10.  [EXP4]       #
#! Activation Function: LeakyReLU    #
#! Optimizer: Adam.                  #
#! Loss func: Cross-Entropy.         #
#! Result: Train_accuracy = 91.682%  #
#           Test_accuracy = 58.120%  #
######################################


# save path
network_save_path = "./cf_lenet_50E_32B_1LR.pth"
graph_save_path = "./result_pic/"

# draw plot
loss_output_step = 200.0
train_set_acc=[]
test_set_acc=[]
loss_val=[]
real_loss_step=[]

# param
EPOCH = 50
BATCH_SIZE = 32
LR = 0.001
IS_LR_DECLINE = False
DOWNLOAD=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,6,5,1,0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5,1,0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )
        self.out = nn.Sequential(
            nn.Linear(5*5*16,120),
            nn.LeakyReLU(),
            nn.Linear(120,84),
            nn.LeakyReLU(),
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
loss_func = nn.CrossEntropyLoss().to(device)
# Dynamic Learing Rate
if IS_LR_DECLINE is True:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,30,40],gamma=0.2)


#  data prepare
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
train_data= torchvision.datasets.CIFAR10(
    root="./cifar-10",
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

test_data= torchvision.datasets.CIFAR10(
    root="./cifar-10",
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
step = 0
for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0

    # choose any frequncy of output you like.
    if epoch%1==0 and epoch != 0:
        # train_data test
        correct_t = 0
        total_t = 0
        for data in train_loader:
            images_t, labels_t = data
            outputs_t = net(images_t.to(device))
            _, predicted_t = torch.max(outputs_t.data, 1)
            total_t += labels_t.size(0)
            correct_t += (predicted_t == labels_t.to(device)).sum().item() 

        print('epoch: %d | Train dataset accuracy: %.3f %%' % (epoch, 100 * correct_t*1.0 / total_t))
        train_set_acc.append(100 * correct_t*1.0 / total_t)

        # test
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()  

        print('epoch: %d | Test accuracy: %.3f %%' % (epoch, 100 * correct * 1.0/ total))
        test_set_acc.append(100 * correct * 1.0/ total)

    for i, data in enumerate(train_loader):
        step+=1
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        output = net(inputs.to(device))
        loss = loss_func(output, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % int(loss_output_step) == 0 and i != 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / loss_output_step))
            loss_val.append(running_loss / loss_output_step)
            real_loss_step.append(step)
            running_loss = 0.0
    if IS_LR_DECLINE is True:
        scheduler.step()
        print(scheduler.get_lr())


print('Finished Training')
torch.save(net.state_dict(), network_save_path)


# final result
correct_t = 0
total_t = 0
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images_t, labels_t = data
        outputs_t = net(images_t.to(device))
        _, predicted_t = torch.max(outputs_t.data, 1)
        total_t += labels_t.size(0)
        correct_t += (predicted_t == labels_t.to(device)).sum().item()  

    print('FINAL | Train dataset accuracy: %.3f %%' % ( 100 * correct_t*1.0 / total_t))
    train_set_acc.append(100 * correct_t * 1.0 / total_t)

    for data in test_loader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item() 
    print('FINAL | Test dataset accuracy: %.3f %%' % (100 * correct / total))
    test_set_acc.append(100 * correct * 1.0 / total)

# plot
x1 = range(EPOCH)
x2 = range(EPOCH)
x3 = real_loss_step # batch_size 和 train_dataset_size 不是整数倍

y1 = train_set_acc
y2 = test_set_acc
y3 = loss_val

plt.title("LeNet-CIFAR10-dataset_accuracy")
plt.plot(x1,y1,'o-b',label='train_acc')
plt.plot(x2,y2,'o-r',label='test_acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(graph_save_path+"cf_lenet_leaky_relu_acc.png")
plt.cla()


plt.title("LeNet-CIFAR10-Loss")
plt.plot(x3,y3,'o-y',label='loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig(graph_save_path+"cf_lenet_leaky_relu_loss.png")