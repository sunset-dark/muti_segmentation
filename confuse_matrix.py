import torch


# 混淆矩阵
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes  # 分类个数(加了背景之后的)
        self.mat = None  # 混淆矩阵

    def update(self, a, b):  # 计算混淆矩阵
        n = self.num_classes
        if self.mat is None:  # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)  # 真正的分类标签(0-n-1)
            inds = n * a[k].to(torch.int64) + b[k]  # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def compute(self):
        h = self.mat.float()

        acc_global = torch.diag(h).sum() / h.sum()  # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        recall = torch.diag(h) / (h.sum(1)+1e-8)           # 计算每个类别的 recall
        precision = torch.diag(h) / (h.sum(0)+1e-8)        # 计算每个类别的 precision
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)+1e-8)  # 计算iou
        return acc_global, recall, precision,iou

    def __str__(self):
        acc_global, recall,precision, iou = self.compute()
        return (
            'global correct: {:.4f}\n'
            'precision: {}\n'
            'recall: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.4f}').format(
            acc_global.item(),          # 像素准确性
            ['{:.4f}'.format(i) for i in precision.tolist()],   # 精确率
            ['{:.4f}'.format(i) for i in recall.tolist()],      # 召回率
            ['{:.4f}'.format(i) for i in iou.tolist()],
            iou.mean().item())
