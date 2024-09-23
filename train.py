import os
import torch
import shutil
import argparse
import math
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler # 学习率衰减
from dataset import MyDataset                   # 自定义的 dataset
import transforms as T
from model import UNet
from utils import (
    compute_gray,           # mask 的灰度值
    train_one_epoch,        # 训练一个 epoch
    evaluate,               # 评价模型精度
    plot,                   # 可视化数据
    plot_lr_decay,          # 学习率下降曲线
    plt_loss_iou            # 训练集 + 测试集的loss和 miou曲线
)


# 训练集预处理
class SegmentationPresetTrain:
    def __init__(self, base_size=600, rcrop_size=480, hflip_prob=0.5, vflip_prob=0.5,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        min_size = int(0.5 * base_size)
        max_size = int(1.5 * base_size)

        trans = [T.RandomResize(min_size, max_size)]  # 随机缩放
        if hflip_prob > 0:  # 水平翻转
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:  # 垂直翻转
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(rcrop_size),         # 随机裁剪,需要保证一个batch 的 size相等
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),  # normalization
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


# 测试集预处理
class SegmentationPresetTest:
    def __init__(self,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),  # normalization
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    # 显示训练的设备
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))

    with open('./run_results/train_log_results.txt', "a") as f:  # 保存训练信息, a --> 在文件中追加信息
        info = f"[train hyper-parameters: {args}]\n"
        f.write(info)

    # 训练集和测试集的预处理
    train_tf = SegmentationPresetTrain(base_size=args.base_size, rcrop_size=args.crop_size)
    test_tf = SegmentationPresetTest()

    # 输出mask灰度值的txt文件
    num_classes = compute_gray()

    # 数据集处理
    trainDataset = MyDataset(imgs_path='./data/train/images',txt_path='./data/grayList.txt', transform=train_tf)
    testDataset = MyDataset(imgs_path='./data/test/images',txt_path='./data/grayList.txt', transform=test_tf)

    # 计算加载的线程数
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % num_workers)

    # 加载数据
    trainLoader =DataLoader(trainDataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    testLoader =DataLoader(testDataset, batch_size=1, num_workers=num_workers, shuffle=False)

    # 可以查看具体的训练图像被预处理成啥样
    dataloader = next(iter(trainLoader))
    plot(data_loader=dataloader)
    return

    model = UNet(num_classes=num_classes)
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-8)

    # 绘制学习率衰减图
    lf_plot = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_plot)
    plot_lr_decay(scheduler,optimizer,args.epochs)       # 学习率衰减图

    # 自适应学习率衰减
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_mean_iou = 0.0
    train_loss_list = []        # 训练集的损失
    test_loss_list = []         # 测试集的损失
    train_miou_list = []        # 训练集的 mean iou
    test_miou_list = []         # 测试集的 mean iou
    for epoch in range(args.epochs):
        # 训练一个epoch ,返回训练集和测试集的损失
        train_loss, test_loss, lr = train_one_epoch(model=model, optim=optimizer, train_loader=trainLoader, test_loader=testLoader, device=device)

        scheduler.step()        # 学习率衰减

        # 评估模型
        train_miou,test_miou,test_confmat= evaluate(model=model, train_loader=trainLoader,test_loader=testLoader, device=device,num=num_classes)

        # 记录训练集和测试集的信息
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_miou_list.append(train_miou)
        test_miou_list.append(test_miou)

        with open('./run_results/train_log_results.txt', "a") as f:  # 保存训练信息, a --> 在文件中追加信息
            info = f"[epoch: {epoch+1}]\n" + test_confmat + '\n\n'
            f.write(info)

        if test_miou > best_mean_iou:   # 保留最好的权重
            best_mean_iou = test_miou
            torch.save(model.state_dict(), './run_results/best_model.pth')

        # 控制台的打印信息
        print("[epoch:%d]"%(epoch+1))
        print("learning rate:%.8f" % lr)
        print("train loss:%.4f \t train mean iou:%.4f"%(train_loss,train_miou))
        print("test loss:%.4f \t test mean iou:%.4f"%(test_loss,test_miou),end='\n\n')

    # 绘制loss和iou曲线
    plt_loss_iou(train_loss_list,test_loss_list,train_miou_list,test_miou_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="unet segmentation")
    parser.add_argument("--base-size",default=400,type=int)               # 根据图像大小更改
    parser.add_argument("--crop-size",default=240,type=int)                # 中心裁剪的尺寸
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lrf',default=0.001,type=float)                  # 最终学习率 = lr * lrf

    args = parser.parse_args()
    print(args)

    # 删除上次保留权重和训练日志，重新创建
    if os.path.exists("./run_results"):
        shutil.rmtree('./run_results')
    os.mkdir("./run_results")

    main(args)
