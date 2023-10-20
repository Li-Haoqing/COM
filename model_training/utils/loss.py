import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):

    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss


def criterion(inputs, target):
    if isinstance(inputs, list):
        losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = F.binary_cross_entropy_with_logits(inputs, target)

    return total_loss


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(size_average=True)
    with torch.autocast(enabled=False, device_type='cuda'):
        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss

