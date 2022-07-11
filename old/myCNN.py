from pickletools import optimize
from random import shuffle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False #第一次打开设置为True，下载文件

#准备训练集
train_data=torchvision.datasets.MNIST(
    root=r"D:\code\minist",#根据自己电脑的位置设置路径
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

#取消注释可以看到训练集数据的图片
#print(train_data.train_data.size())
#print(train_data.train_labels.size())
#plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
#plt.title('%i' %train_data.train_labels[0])
#plt.show()


train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

#准备测试集
test_data=torchvision.datasets.MNIST(
    root=r"D:\code\minist",
    train=False,
    transform=torchvision.transforms.ToTensor(),#下载的数据改为Tensor形式
    download=False
)

test_x=Variable(torch.unsqueeze(test_data.test_data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255
#
test_y=test_data.test_labels[:2000]

#建立CNN神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),#卷积层
            nn.ReLU(),#激活函数
            nn.MaxPool2d(kernel_size=2),#池化层
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out=nn.Linear(32*7*7,10)

    #三维数据展平
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output

cnn=CNN()
#打印结构
#print(cnn)


#优化器
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
#交叉熵损失函数
loss_func=nn.CrossEntropyLoss()

#训练过程
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = Variable(b_x)
        b_y = Variable(b_y)

        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50 == 0:
            test_output=cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = float((pred_y == test_y).numpy().astype(int).sum())/float(test_y.size(0))

            print('Epoch:',epoch,'| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

#取前十个数据测试效果
test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
#打印测试结果
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')