"""
@Author: Du Yunhao
@Filename: loss.py
@Contact: dyh_bupt@163.com
@Time: 2021/8/12 11:17
@Discription: loss funcion
"""
import torch
from torch import nn

class BCE_VIRAT(nn.Module):
    def __init__(self, reduction="mean", hard_thres=-1):
        """
        :param hard_thres:
            -1：软标签损失，直接基于标注中的软标签计算BECLoss；
            >0：硬标签损失，将标签大于hard_thres的置为1，否则为0；
        """
        super(BCE_VIRAT, self).__init__()
        self.hard_thres = hard_thres
        self._loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, x, y):
        if self.hard_thres > 0:  # 硬标签
            mask = y > self.hard_thres
            y[mask] = 1.
            y[~mask] = 0.
        # weight = torch.tensor([2.0869271159324994, 1.0968095583318394, 4.667857504911766, 1.6595608352187452, 3.6011781840303687, 2.6403159830224547, 4.869071729774468], device=x.device)
        # pos_weight = torch.tensor([1.954460531501126, 0.6904418649003278, 4.658420747351811, 1.4485650738429943, 3.5735073030418634, 2.5663047647752473, 4.861361591348501], device=x.device)
        # _loss_fn = nn.BCEWithLogitsLoss(reduction="mean", weight=weight, pos_weight=pos_weight)
        loss = self._loss_fn(x, y)
        # loss = _loss_fn(x, y)
        return loss

if __name__ == '__main__':
    label = torch.tensor([
        [0, 1, .5],
        [1, 0, .5]
    ], dtype=torch.float64)
    pred = torch.tensor([
        [-1000, 1000, 0],
        [1000, -1000, 0]
    ], dtype=torch.float32, requires_grad=True)
    loss_fn = BCE_VIRAT(hard_thres=0.6)
    loss = loss_fn(pred, label)
    loss.backward()
    print(loss)
    print(pred.grad)
