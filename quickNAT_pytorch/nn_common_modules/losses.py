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


class CalDiceLoss(_WeightedLoss):
    """
        Dice Loss for a batch of samples
        """
    # @torch.jit.script_method
    def forward(self, output, target, weights=None, ignore_index=None):
        """
        Forward pass
        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :return:
        """
        '''
            TPU = tf.reduce_sum(phi(yPred)*yTrue)
            TPD = tf.reduce_sum(yTrue)
            FPD = tf.reduce_sum((1-phi(yPred))*yTrue)
            FND = tf.reduce_sum((1-phi(-yPred))*(1-yTrue))
            loss = 1-2*TPU/(2*TPD+FPD+FND)
            1-2*TPU/(2*TPD+FPD+FND)
        '''
        phi = lambda t: 1.0+F.log_softmax(torch.min(t,t/3.0),dim=1)/torch.log(torch.tensor(2.0))
        eps = 0.0001
        encoded_target = output.detach() * 0
        dim = len(output.shape) - 2
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        if weights is None:
            weights = 1
        else:
            batch_dim = weights.size()[0]
            weights = weights.sum(0) / batch_dim
        intersection = phi(output) * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1)
        if dim == 2:
            numerator = 2 * numerator
        elif dim == 3:
            numerator = 2 * numerator.sum(1)
        else:
            raise ValueError('Expected dimension 2 or 3, got dim {}'.format(
                dim))
        #denominator = output + encoded_target
        TPD = encoded_target
        FPD = (1.0 - phi(output)) * encoded_target
        FND = (1.0 - phi(-1.0*output)) * (1.0 - encoded_target)
        denominator = 2*TPD + FPD+FND
        # if ignore_index is not None:
        #    denominator[mask] = 0
        if dim == 2:
            denominator = denominator.sum(0).sum(1).sum(1) + eps
        elif dim == 3:
            denominator = denominator.sum(0).sum(1).sum(1).sum(1) + eps
        else:
            raise ValueError('Expected dimension 2 or 3, got dim {}'.format(
                dim))
        dice = numerator / denominator

        loss_per_channel = 1.0-dice
        return loss_per_channel.sum() / output.size(1), dice


class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weight=None):
        output = F.softmax(output, dim=1)
        target = target.unsqueeze(1)
        encoded_target = torch.zeros_like(output)
        encoded_target = encoded_target.scatter(1, target, 1)

        intersection = output * encoded_target
        intersection = intersection.sum(2).sum(2)

        num_union_pixels = output + encoded_target
        num_union_pixels = num_union_pixels.sum(2).sum(2)
        loss_per_class = 1 - ((2 * intersection + 1) / (num_union_pixels + 1))
        if weight is None:
            weight = torch.ones_like(loss_per_class)
        loss_per_class *= weight

        return (loss_per_class.sum(1) / (num_union_pixels != 0).sum(1).float()).mean()


class CombinedLoss(_Loss):
    """
    A combination of dice and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.dice_loss = DiceLoss()
        self.cal_dice_loss = CalDiceLoss()

    def forward(self, input, target, class_weights=None, dice_weights=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param class_weights: torch.tensor (NxHxW)
        :return: scalar
        """
        y_2 = self.dice_loss(input, target, dice_weights)
        if class_weights is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), class_weights))

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
