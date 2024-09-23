import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet


def main():
    weights_path = "./run_results/best_model.pth"
    test_path = './inference'

    with open('./data/grayList.txt','r') as f:
        gray = f.read().splitlines()

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = UNet(num_classes=len(gray))
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    model.eval()  # 进入验证模式

    # inference 所有图片路径
    test_imgs = [os.path.join(test_path, i) for i in os.listdir(test_path)]

    # load image
    for test_img in test_imgs:
        original_img = Image.open(test_img).convert('RGB')
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        img = data_transform(original_img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = model(img.to(device))

            prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # 将灰度值映射回去
            for i in np.unique(prediction):
                prediction[prediction==i] = gray[i]

            a = test_img.split('.')[-2]
            save_path ='.'+a + '_result.' + test_img.split('.')[-1]

            plt.imsave(save_path,prediction,cmap='gray')    # 保存图像
            # 打印提示信息
            print(f"图像 {test_img} 已处理并保存为 {save_path}")
    print("所有图像处理完成。")

if __name__ == '__main__':
    main()
