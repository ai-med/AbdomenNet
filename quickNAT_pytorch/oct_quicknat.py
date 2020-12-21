import numpy as np
import torch
import torch.nn as nn
from nn_common_modules import modules as sm
from models.unet.octconv import OctaveConv


class OctaveQuickNat(nn.Module):
    """
    A PyTorch implementation of QuickNAT
    """

    def __init__(self, params, alpha_in=0, alpha_out=0):
        """
        :param params: {'num_channels':1,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_classes':28
                        'se_block': False,
                        'drop_out':0.2}
        """
        super(OctaveQuickNat, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.conv1 = OctaveConv(in_channels=params['num_channels'], out_channels=params['num_filters'],
                                kernel_size=(
                                    params['kernel_h'], params['kernel_w']),
                                padding=(padding_h, padding_w),
                                stride=params['stride_conv'],
                                alpha_in=alpha_in)
        params['num_channels'] = params['num_filters']
        self.encode1 = OctaveEncoderBlock(params)
        self.encode2 = OctaveEncoderBlock(params)
        self.encode3 = OctaveEncoderBlock(params)
        self.encode4 = OctaveEncoderBlock(params)
        self.bottleneck = OctaveDenseBlock(params)
        params['num_channels'] = params['num_filters'] * 2
        self.decode1 = OctaveDecoderBlock(params)
        self.decode2 = OctaveDecoderBlock(params)
        self.decode3 = OctaveDecoderBlock(params)
        self.decode4 = OctaveDecoderBlock(params)
        params['num_channels'] = params['num_filters']
        self.out1 = OctaveConv(in_channels=params['num_channels'], out_channels=params['num_channels'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'],
                               alpha_out=alpha_out)
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        """
        :param input: X
        :return: probabiliy map
        """
        input = self.conv1(input)
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)

        bn = self.bottleneck.forward(e4)

        d4 = self.decode4.forward(bn, out4, ind4)
        d3 = self.decode1.forward(d4, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        out, _ = self.out1(d1)
        prob = self.classifier.forward(out)

        return prob


class OctaveDenseBlock(nn.Module):
    """Block with dense connections
    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params, alpha_in=0.5, alpha_out=0.5):
        super(OctaveDenseBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = OctaveConv(in_channels=params['num_channels'], out_channels=params['num_filters'],
                                kernel_size=(
                                    params['kernel_h'], params['kernel_w']),
                                padding=(padding_h, padding_w),
                                stride=params['stride_conv'],
                                alpha_in=alpha_in, alpha_out=alpha_out)
        self.conv2 = OctaveConv(in_channels=conv1_out_size, out_channels=params['num_filters'],
                                kernel_size=(
                                    params['kernel_h'], params['kernel_w']),
                                padding=(padding_h, padding_w),
                                stride=params['stride_conv'],
                                alpha_in=alpha_out, alpha_out=alpha_out)
        self.conv3 = OctaveConv(in_channels=conv2_out_size, out_channels=params['num_filters'],
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                stride=params['stride_conv'],
                                alpha_in=alpha_out, alpha_out=alpha_out)

        self.batchnorm1_h = None if alpha_in == 1 else nn.BatchNorm2d(
            num_features=int(params['num_channels'] * (1 - alpha_in)))
        self.batchnorm1_l = None if alpha_in == 0 else nn.BatchNorm2d(
            num_features=int(params['num_channels'] * alpha_in))

        self.batchnorm2_h = None if alpha_out == 1 else nn.BatchNorm2d(
            num_features=int(conv1_out_size * (1 - alpha_out)))
        self.batchnorm2_l = None if alpha_out == 0 else nn.BatchNorm2d(num_features=int(conv1_out_size * alpha_out))

        self.batchnorm3_h = None if alpha_out == 1 else nn.BatchNorm2d(
            num_features=int(conv2_out_size * (1 - alpha_out)))
        self.batchnorm3_l = None if alpha_out == 0 else nn.BatchNorm2d(num_features=int(conv2_out_size * alpha_out))

        self.prelu_h = nn.PReLU()
        self.prelu_l = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """
        input_h, input_l = input
        o1_h = self.batchnorm1_h(input_h)
        o1_l = self.batchnorm1_l(input_l)
        o2_h = self.prelu_h(o1_h)
        o2_l = self.prelu_l(o1_l)
        o3_h, o3_l = self.conv1((o2_h, o2_l))
        o4_h = torch.cat((input_h, o3_h), dim=1)
        o4_l = torch.cat((input_l, o3_l), dim=1)
        o5_h = self.batchnorm2_h(o4_h)
        o5_l = self.batchnorm2_l(o4_l)
        o6_h = self.prelu_h(o5_h)
        o6_l = self.prelu_l(o5_l)
        o7_h, o7_l = self.conv2((o6_h, o6_l))
        o8_h = torch.cat((input_h, o3_h, o7_h), dim=1)
        o8_l = torch.cat((input_l, o3_l, o7_l), dim=1)
        o9_h = self.batchnorm3_h(o8_h)
        o9_l = self.batchnorm3_l(o8_l)
        o10_h = self.prelu_h(o9_h)
        o10_l = self.prelu_l(o9_l)
        out_h, out_l = self.conv3((o10_h, o10_l))
        return out_h, out_l


class OctaveEncoderBlock(OctaveDenseBlock):
    """Dense encoder block with maxpool and an optional SE block
    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
    """

    def __init__(self, params, alpha_in=0.5, alpha_out=0.5):
        super(OctaveEncoderBlock, self).__init__(params, alpha_in=alpha_in, alpha_out=alpha_out)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
        """

        out_block_h, out_block_l = super(OctaveEncoderBlock, self).forward(input)

        if self.drop_out_needed:
            out_block_h = self.drop_out(out_block_h)
            out_block_l = self.drop_out(out_block_l)

        out_encoder_h, indices_h = self.maxpool(out_block_h)
        out_encoder_l, indices_l = self.maxpool(out_block_l)
        return (out_encoder_h, out_encoder_l), (out_block_h, out_block_l), (indices_h, indices_l)


class OctaveDecoderBlock(OctaveDenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections and SE block
    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, alpha_in=0.5, alpha_out=0.5):
        super(OctaveDecoderBlock, self).__init__(params, alpha_in=alpha_in, alpha_out=alpha_out)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None, weights=None):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """
        input_h, input_l = input
        out_block_h, out_block_l = out_block
        indices_h, indices_l = indices

        if indices is not None:
            unpool_h = self.unpool(input_h, indices_h, out_block_h.shape)
            unpool_l = self.unpool(input_l, indices_l, out_block_l.shape)
        else:
            # TODO: Implement Conv Transpose
            print("You have to use Conv Transpose")

        if out_block is not None:
            concat_h = torch.cat((out_block_h, unpool_h), dim=1)
            concat_l = torch.cat((out_block_l, unpool_l), dim=1)
        else:
            concat_h = unpool_h
            concat_l = unpool_l
        out_block_h, out_block_l = super(OctaveDecoderBlock, self).forward((concat_h, concat_l))

        if self.drop_out_needed:
            out_block_h = self.drop_out(out_block_h)
            out_block_l = self.drop_out(out_block_l)
        return out_block_h, out_block_l
