import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,imgs_path,txt_path,transform=None):
        self.imgs = [os.path.join(imgs_path,i) for i in os.listdir(imgs_path)]
        self.transform = transform

        txt = open(txt_path,'r')
        self.txt = txt.read().splitlines()      # ['0', '1', '4', '9', '12', '15', '19', '20', '255']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.imgs[index]                           # 随机读取一张训练集图片
        mask_path = image_path.replace('images','masks')        # 训练集图片对应的标签图片
        '''
        image 和 label 名字不是完全一样需要更换（例如图片后缀不同等等），用下列代码替换，示例
        mask_path = image_path.replace('.jpg','.png')
        '''
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)

        # 将多分割灰度值赋值为 0 1 2 ...
        for gray in np.unique(mask):
            mask[mask==gray] = self.txt.index(str(gray))
        mask = Image.fromarray(mask)

        if self.transform is not None:              # 预处理
            image,mask = self.transform(image,mask)

        return image, mask
