import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms, datasets, models
import time
import os
import cv2
import numpy as np
from unetplus import UNet_3Plus
from matplotlib import pyplot as plt
from unetplus import *

device = torch.device('cuda')
class MYDATA(Dataset):
    def __init__(self, path):
        imgpath=path+'/imgs/'
        labelpath=path+'/masks/'

        imgs_path=os.listdir(imgpath)
        self.imgs=[imgpath+x for x in imgs_path]

        label_path=os.listdir(labelpath)
        self.labels = [labelpath + x for x in label_path]

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index],0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.array(img).reshape(128, 128,3)
        img = img / 255.
        img = np.transpose(img, [2, 0, 1])


        labels = cv2.imread(self.labels[index],0)
        labels = cv2.cvtColor(labels, cv2.COLOR_GRAY2RGB)
        labels = np.array(labels).reshape(128, 128,3)
        labels = labels / 255.
        labels = np.transpose(labels, [2, 0, 1])
        # print(self.imgs[index])
        # print(self.labels[index])
        # print('********************')

        #print(img.shape)
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(labels,dtype=torch.float32)

        return img, label

    def __len__(self):
        return len(self.imgs)



train_set=MYDATA('data/train')
test_set=MYDATA('data/test')
print(len(train_set))

train_iter = DataLoader(train_set, batch_size=32)
test_iter = DataLoader(test_set, batch_size=32)


mynet =UNet_3Plus().to(device)


train_loss_list=[]
test_loss_list=[]
num_epochs=200
lr = 0.0001
min_loss=10
loss_func= torch.nn.MSELoss()
optimizer = optim.Adam(mynet.parameters(),lr=lr,weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 调整学习率
#net = mynet.train()
print("training on", device)

for epoch in range(num_epochs):
    start = time.time()
    #mynet.to(device)
    mynet.train()  # 训练模式
    train_loss_sum, batch_count = 0.0, 0.0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()  # 梯度清零
        y_hat = mynet(X)
        loss = torch.sqrt(loss_func(y_hat, y))
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        batch_count += 1
        print('epoch{} batch_count {}  batch_loss {} '.format(epoch+1,batch_count,train_loss_sum/batch_count))

    test_loss_sum, test_batch_count = 0.0, 0.0
    with torch.no_grad():
        mynet.eval()  # 评估模式

        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = mynet(X)
            tess_loss = torch.sqrt(loss_func(y_hat, y))

            test_loss_sum += tess_loss.item()
            test_batch_count += 1

    print('********epoch{} train_loss{} test_loss{} time{}'.format(epoch + 1, train_loss_sum / batch_count,test_loss_sum / test_batch_count, time.time() - start))
    print('***********************************************')
    train_loss_list.append(train_loss_sum / batch_count)
    test_loss_list.append(test_loss_sum / test_batch_count)
    val_loss=test_loss_sum / test_batch_count
    scheduler.step()
    if val_loss<min_loss:
        min_loss=val_loss
        torch.save(mynet.state_dict(),'Unet3plusmodel.pkl')

plt.figure()
plt.xlabel('Epoch') #为x轴命名为“x”
plt.ylabel('RMSE') #为y轴命名为“y”
plt.plot([x+1 for x in range(num_epochs)],train_loss_list)
plt.plot([x+1 for x in range(num_epochs)],test_loss_list)
plt.legend(['y = Training','y = Validation']) #打出图例
plt.savefig('loss.jpg')
plt.show()