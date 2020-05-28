"""
Description
++++++++++++++++++++++
Addition losses module defines classses which are commonly used particularly in segmentation and are not part of standard pytorch library.

Usage
++++++++++++++++++++++
Import the package and Instantiate any loss class you want to you::

    from nn_common_modules import losses as additional_losses
    loss = additional_losses.DiceLoss()

    Note: If you use DiceLoss, insert Softmax layer in the architecture. In case of combined loss, do not put softmax as it is in-built

Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.autograd import Variable


class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target):
        #print('output: ', output.shape)
        #print('target: ', target.shape)
        output = F.softmax(output, dim=1)
        #print(output.shape)
        eps = 0.0001
        target = target.unsqueeze(1)
        #print(target.shape)
        #print(torch.min(target))
        #print(torch.max(target))
        encoded_target = torch.zeros_like(output)
        #print(encoded_target)

        encoded_target = encoded_target.scatter(1, target, 1)
        #print(encoded_target)
        intersection = output * encoded_target
        intersection = intersection.sum(2).sum(2)
        #print(intersection)

        num_union_pixels = output + encoded_target
        num_union_pixels = num_union_pixels.sum(2).sum(2)
        #print(num_union_pixels)

        loss_per_class = 1 - ((2 * intersection) / (num_union_pixels + eps))

        return (loss_per_class.sum(1) / (num_union_pixels != 0).sum(1).float()).mean()


class CombinedLoss(_Loss):
    """
    A combination of dice and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.dice_loss = DiceLoss()

    def forward(self, input, target, class_weights=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param class_weights: torch.tensor (NxHxW)
        :return: scalar
        """
        y_2 = self.dice_loss(input, target)
        #print('dice loss', y_2)
        if class_weights is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), class_weights))
        #print('ce loss: ', y_1)
        return y_1 + y_2


class CombinedLoss_KLdiv(_Loss):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss_KLdiv, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=None):
        """
        Forward pass

        """
        input, kl_div_loss = input
        # input_soft = F.softmax(input, dim=1)
        y_2 = torch.mean(self.dice_loss(input, target))
        if weight is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        return y_1, y_2, kl_div_loss


# Credit to https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    """

    def __init__(self, gamma=2, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """Forward pass

        :param input: shape = NxCxHxW
        :type input: torch.tensor
        :param target: shape = NxHxW
        :type target: torch.tensor
        :return: loss value
        :rtype: torch.tensor
        """

        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
