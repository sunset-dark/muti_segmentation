import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import confuse_matrix
from PIL import Image


# 获取多分割的前景像素点，并保存在txt文件中
def compute_gray():
    root = './data/train/masks'             # 训练mask的路径
    masks_path = [os.path.join(root,i) for i in os.listdir(root)]
    gray = []           # 前景像素点
    for i in tqdm(masks_path,desc="gray compute"):
        img = Image.open(i)
        img = np.unique(img)        # 获取mask的灰度值
        for j in img:
            if j not in gray:
                gray.append(j)

    with open('./data/grayList.txt','w') as f:
        gray.sort()     # 灰度值从小到大排序
        print('mask gray is:',gray)
        print('unet output is:',len(gray))
        for i in gray:
            f.write(str(i))
            f.write('\n')
    return len(gray)


# 训练一个 epoch
def train_one_epoch(model, optim,train_loader,test_loader, device):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)   # 定义交叉熵损失函数

    model.train()
    train_running_loss = 0.0        # 训练集的总损失
    train_num = 0                   # 训练集的总数
    for train_image, train_target in tqdm(train_loader):
        train_image, train_target = train_image.to(device), train_target.to(device)

        train_output = model(train_image)               # 前向传播
        loss = criterion(train_output, train_target)    # 计算损失
        optim.zero_grad()                   # 梯度清零
        loss.backward()                     # 反向传播
        optim.step()                        # 梯度更新

        train_num += train_image.size(0)
        train_running_loss += loss.item()

    # 计算在测试集上的损失
    model.eval()
    test_running_loss = 0.0         # 测试集的总损失
    test_num = 0                    # 测试集的总数
    with torch.no_grad():
        for test_image, test_target in tqdm(test_loader):
            test_image, test_target = test_image.to(device), test_target.to(device)

            test_output = model(test_image)               # 前向传播
            loss = criterion(test_output, test_target)    # 计算损失

            test_num += test_image.size(0)
            test_running_loss += loss.item()

    lr = optim.param_groups[0]["lr"]
    return train_running_loss/train_num, test_running_loss/test_num,lr


# 计算性能指标
def evaluate(model, train_loader, test_loader,device,num):

    model.eval()
    train_confmat = confuse_matrix.ConfusionMatrix(num_classes=num)         # 评估训练集的混淆矩阵
    test_confmat = confuse_matrix.ConfusionMatrix(num_classes=num)          # 评估测试集的混淆矩阵

    with torch.no_grad():
        for train_image, train_target in tqdm(train_loader):        # 计算在训练集的精度
            train_image, train_target = train_image.to(device), train_target.to(device)

            train_output = model(train_image)
            train_output = torch.argmax(train_output, dim=1)
            train_confmat.update(train_target, train_output)
        train_miou = float(str(train_confmat)[-6:])    # 取出 miou

        for test_image, test_target in tqdm(test_loader):        # 计算在测试集的精度
            test_image, test_target = test_image.to(device), test_target.to(device)

            test_output = model(test_image)
            test_output = torch.argmax(test_output, dim=1)
            test_confmat.update(test_target, test_output)
        test_miou = float(str(test_confmat)[-6:])   # 取出 miou

    return train_miou,test_miou,str(test_confmat)


# 可视化数据,只展示2个
def plot(data_loader,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    plt.figure(figsize=(12, 8))
    imgs, labels = data_loader
    print('images:',imgs.shape,imgs.dtype)                  # torch.Size([batch, 3, 96, 96]) torch.float32
    print('labels:',labels.shape,labels.dtype)              # torch.Size([batch, 96, 96]) torch.int64
    print('classes:',np.unique(labels))                      # 0 1 255 只包含 0 1 2...255(255为预处理填充的部分)

    for i, (x, y) in enumerate(zip(imgs[:2], labels[:2])):
        x = np.transpose(x.numpy(), (1, 2, 0))
        x[:, :, 0] = x[:, :, 0] * std[0] + mean[0]  # 去 normalization
        x[:, :, 1] = x[:, :, 1] * std[1] + mean[1]
        x[:, :, 2] = x[:, :, 2] * std[2] + mean[2]
        y = y.numpy()

        plt.subplot(2, 2, i + 1)
        plt.imshow(x)

        plt.subplot(2, 2, i + 3)
        plt.imshow(y)
    plt.show()


# 绘制学习率衰减过程
def plot_lr_decay(scheduler,optimizer,epochs):
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LambdaLR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('./run_results/LR_decay.png', dpi=300)


# 绘制loss 和 iou曲线
def plt_loss_iou(train_loss,test_loss,train_iou,test_iou):
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label='train loss',linestyle='-',color='g')
    plt.plot(test_loss,label='test loss',linestyle='-.',color='r')
    plt.title('loss curve')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_iou,label='train mean iou',linestyle='-',color='g')
    plt.plot(test_iou,label='test mean iou',linestyle='-.',color='r')
    plt.title('mean iou curve')
    plt.legend()

    plt.savefig('./run_results/loss_iou_curve.png',dpi=300)
