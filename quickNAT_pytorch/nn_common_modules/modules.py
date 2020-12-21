"""
Description
++++++++++++++++++++++
Building blocks of segmentation neural network

Usage
++++++++++++++++++++++
Import the package and Instantiate any module/block class you want to you::

    from nn_common_modules import modules as additional_modules
    dense_block = additional_modules.DenseBlock(params, se_block_type = 'SSE')

Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
from squeeze_and_excitation import squeeze_and_excitation as se
import torch.nn.functional as F
from math import ceil,floor

class DenseBlock(nn.Module):
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

    def __init__(self, params, se_block_type=None):
        super(DenseBlock, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
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

        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class EncoderBlock(DenseBlock):
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

    def __init__(self, params, se_block_type=None):
        super(EncoderBlock, self).__init__(params, se_block_type=se_block_type)
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

        out_block = super(EncoderBlock, self).forward(input)
        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(DenseBlock):
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

    def __init__(self, params, se_block_type=None):
        super(DecoderBlock, self).__init__(params, se_block_type=se_block_type)
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
        if indices is not None:
            unpool = self.unpool(input, indices)
        else:
            # TODO: Implement Conv Transpose
            print("You have to use Conv Transpose")

        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(DecoderBlock, self).forward(concat)

        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


class ClassifierBlock(nn.Module):
    """
    Last layer

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_c':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(
            params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights for classifier regression, defaults to None
        :type weights: torch.tensor (N), optional
        :return: logits
        :rtype: torch.tensor
        """
        batch_size, channel, a, b = input.size()
        if weights is not None:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out_conv = F.conv2d(input, weights)
        else:
            out_conv = self.conv(input)
        return out_conv


class GenericBlock(nn.Module):
    """
    Generic parent class for a conv encoder/decoder block.

    :param params: {'kernel_h': 5
                        'kernel_w': 5
                        'num_channels':64
                        'num_filters':64
                        'stride_conv':1
                        }
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional    
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(GenericBlock, self).__init__()
        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']
        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.prelu = nn.PReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Custom weights for convolution, defaults to None
        :type weights: torch.tensor [FloatTensor], optional
        :return: [description]
        :rtype: [type]
        """

        _, c, h, w = input.shape
        if weights is None:
            x1 = self.conv(input)
        else:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(self.out_channel, c, 1, 1)
            x1 = F.conv2d(input, weights)
        x2 = self.prelu(x1)
        x3 = self.batchnorm(x2)
        return x3


class SDnetEncoderBlock(GenericBlock):
    """
    A standard conv -> prelu -> batchnorm-> maxpool block without dense connections

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

    def __init__(self, params, se_block_type=None):
        super(SDnetEncoderBlock, self).__init__(params, se_block_type)
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

        out_block = super(SDnetEncoderBlock, self).forward(input, weights)

        if self.SELayer:
            out_block = self.SELayer(out_block, weights)
        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class SDnetDecoderBlock(GenericBlock):
    """Standard decoder block with maxunpool -> skipconnections -> conv -> prelu -> batchnorm, without dense connections and an optional SE blocks

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

    def __init__(self, params, se_block_type=None):
        super(SDnetDecoderBlock, self).__init__(params, se_block_type)
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
        :return: Forward pass
        :rtype: torch.tensor
        """

        unpool = self.unpool(input, indices)
        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(SDnetDecoderBlock, self).forward(concat, weights)
        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


class SDNetNoBNEncoderBlock(nn.Module):
    """
     Encoder Block for Bayesian Network
    """

    def __init__(self, params):
        super(SDNetNoBNEncoderBlock, self).__init__()
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']
        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.relu(x1)
        out_encoder, indices = self.maxpool(x2)
        return out_encoder, x2, indices


class SDNetNoBNDecoderBlock(nn.Module):
    """
     Decoder Block for Bayesian Network
    """

    def __init__(self, params):
        super(SDNetNoBNDecoderBlock, self).__init__()
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']

        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.relu = nn.ReLU()

        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None):
        unpool = self.unpool(input, indices)
        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        x1 = self.conv(concat)
        x2 = self.relu(x1)
        return x2

class FSDenseBlock(nn.Module):
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

    def __init__(self, params, se_block_type=None):
        #TODO: params reduction ratio, kernel_d
        super(FSDenseBlock, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'], reduction_ratio=2)

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'], reduction_ratio=2)


        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])
        
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_filters'])
        conv2_out_size = int(params['num_filters'] )

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w'] ),
                               padding=(padding_h, padding_w ),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w'] ),
                               padding=(padding_h, padding_w ),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.prelu(input)
        o2 = self.conv1(o1)
        o3 = self.batchnorm1(o2)
        o4 = torch.max(input, o3)

        o5 = self.prelu(o4)
        o6 = self.conv2(o5)
        o7 = self.batchnorm2(o6)
        o8 = torch.max(o4,o7)

        o9 = self.prelu(o8)
        o10 = self.conv3(o9)
        out = self.batchnorm3(o10)
        return out

class FSInputBlock(nn.Module):
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

    def __init__(self, params, se_block_type=None):
        # TODO: params['reduction_ratio'], kernel_d
        super(FSInputBlock, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'], reduction_ratio=2)

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'], reduction_ratio=2)


        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_filters'])
        conv2_out_size = int( params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w'] ),
                               padding=(padding_h, padding_w ),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w'] ),
                               padding=(padding_h, padding_w ),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm0 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout(params['drop_out'])
        else:
            self.drop_out_needed = False

        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.batchnorm0(input)
        o2 = self.conv1(o1)
        o3 = self.batchnorm1(o2)
        o4 = torch.max(input, o3)

        o5 = self.prelu(o4)
        o6 = self.conv2(o5)
        o7 = self.batchnorm2(o6)
        o8 = torch.max(o4, o7)

        o9 = self.prelu(o8)
        o10 = self.conv3(o9)
        out_block = self.batchnorm3(o10)

        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class FSEncoderBlock(FSDenseBlock):
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

    def __init__(self, params, se_block_type=None):
        super(FSEncoderBlock, self).__init__(params, se_block_type=se_block_type)
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

        out_block = super(FSEncoderBlock, self).forward(input)
        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices

class FSDecoderBlock(FSDenseBlock):
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

    def __init__(self, params, se_block_type=None, split=False):
        super(FSDecoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])
        self.split=split

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
        if indices is not None:
            unpool = self.unpool(input, indices)
        else:
            # TODO: Implement Conv Transpose
            print("You have to use Conv Transpose")

        '''
        diffx = out_block.size()[4] - unpool.size()[4]
        diffy = out_block.size()[3] - unpool.size()[3]
        diffz = out_block.size()[2] - unpool.size()[2]

        unpool = F.pad(unpool,
                             (ceil(diffx / 2.0), floor(diffx / 2.0), ceil(diffy  / 2.0), floor(diffy / 2.0),
                              ceil(diffz / 2), floor(diffz / 2.0)))
        '''
        if out_block is not None:

            out_block = torch.max(out_block, unpool)
        else:
            out_block = unpool
        out_block = super(FSDecoderBlock, self).forward(out_block)
        if self.split:
            if self.SELayer:
                self.SELayer = self.SELayer.cuda(1)
            if self.drop_out_needed:
                self.drop_out = self.drop_out.cuda(1)

        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class OctConv2d(nn.modules.conv._ConvNd):
    """Unofficial implementation of the Octave Convolution in the "Drop an Octave" paper.
    oct_type (str): The type of OctConv you'd like to use. ['first', 'A'] both stand for the the first Octave Convolution.
                    ['last', 'C'] both stand for th last Octave Convolution. And 'regular' stand for the regular ones.
    """
    
    def __init__(self, oct_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, alpha_in=0.5, alpha_out=0.5):
        
        if oct_type not in ('regular', 'first', 'last', 'A', 'C'):
            raise InvalidOctType("Invalid oct_type was chosen!")

        oct_type_dict = {'first': (0, alpha_out), 'A': (0, alpha_out), 'last': (alpha_in, 0), 'C': (alpha_in, 0), 
                         'regular': (alpha_in, alpha_out)}        

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        # TODO: Make it work with any padding
        padding = _pair(int((kernel_size[0] - 1) / 2))
        # padding = _pair(padding)
        dilation = _pair(dilation)
        super(OctConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), 1, bias, padding_mode='zeros')

        # Get alphas from the oct_type_dict
        self.oct_type = oct_type
        self.alpha_in, self.alpha_out = oct_type_dict[self.oct_type]
        
        self.num_high_in_channels = int((1 - self.alpha_in) * in_channels)
        self.num_low_in_channels = int(self.alpha_in * in_channels)
        self.num_high_out_channels = int((1 - self.alpha_out) * out_channels)
        self.num_low_out_channels = int(self.alpha_out * out_channels)

        self.high_hh_weight = self.weight[:self.num_high_out_channels, :self.num_high_in_channels, :, :].clone()
        self.high_hh_bias = self.bias[:self.num_high_out_channels].clone()

        self.high_hl_weight = self.weight[self.num_high_out_channels:, :self.num_high_in_channels, :, :].clone()
        self.high_hl_bias = self.bias[self.num_high_out_channels:].clone()

        self.low_lh_weight = self.weight[:self.num_high_out_channels, self.num_high_in_channels:, :, :].clone()
        self.low_lh_bias = self.bias[:self.num_high_out_channels].clone()

        self.low_ll_weight = self.weight[self.num_high_out_channels:, self.num_high_in_channels:, :, :].clone()
        self.low_ll_bias = self.bias[self.num_high_out_channels:].clone()

        self.high_hh_weight.data, self.high_hl_weight.data, self.low_lh_weight.data, self.low_ll_weight.data = \
        self._apply_noise(self.high_hh_weight.data), self._apply_noise(self.high_hl_weight.data), \
        self._apply_noise(self.low_lh_weight.data), self._apply_noise(self.low_ll_weight.data)

        self.high_hh_weight, self.high_hl_weight, self.low_lh_weight, self.low_ll_weight = \
        nn.Parameter(self.high_hh_weight), nn.Parameter(self.high_hl_weight), nn.Parameter(self.low_lh_weight), nn.Parameter(self.low_ll_weight)

        self.high_hh_bias, self.high_hl_bias, self.low_lh_bias, self.low_ll_bias = \
        nn.Parameter(self.high_hh_bias), nn.Parameter(self.high_hl_bias), nn.Parameter(self.low_lh_bias), nn.Parameter(self.low_ll_bias)
        

        self.avgpool = nn.AvgPool2d(2)
 
    def forward(self, x):
        if self.oct_type in ('first', 'A'):
            high_group, low_group = x[:, :self.num_high_in_channels, :, :], x[:, self.num_high_in_channels:, :, :]
        else:
            high_group, low_group = x

        high_group_hh = F.conv2d(high_group, self.high_hh_weight, self.high_hh_bias, self.stride,
                        self.padding, self.dilation, self.groups)
        high_group_pooled = self.avgpool(high_group)

        if self.oct_type in ('first', 'A'):
            high_group_hl = F.conv2d(high_group_pooled, self.high_hl_weight, self.high_hl_bias, self.stride,
                        self.padding, self.dilation, self.groups)
            high_group_out, low_group_out = high_group_hh, high_group_hl

            return high_group_out, low_group_out

        elif self.oct_type in ('last', 'C'):
            low_group_lh = F.conv2d(low_group, self.low_lh_weight, self.low_lh_bias, self.stride,
                            self.padding, self.dilation, self.groups)
            low_group_upsampled = F.interpolate(low_group_lh, scale_factor=2)
            high_group_out = high_group_hh + low_group_upsampled

            return high_group_out

        else:
            high_group_hl = F.conv2d(high_group_pooled, self.high_hl_weight, self.high_hl_bias, self.stride,
                        self.padding, self.dilation, self.groups)
            low_group_lh = F.conv2d(low_group, self.low_lh_weight, self.low_lh_bias, self.stride,
                            self.padding, self.dilation, self.groups)
            low_group_upsampled = F.interpolate(low_group_lh, scale_factor=2)
            low_group_ll = F.conv2d(low_group, self.low_ll_weight, self.low_ll_bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
            high_group_out = high_group_hh + low_group_upsampled
            low_group_out = high_group_hl + low_group_ll

        return high_group_out, low_group_out

    @staticmethod
    def _apply_noise(tensor, mu=0, sigma=0.0001):
        noise = torch.normal(mean=torch.ones_like(tensor) * mu, std=torch.ones_like(tensor) * sigma)

        return tensor + noise


class OctReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu_h, self.relu_l = nn.ReLU(inplace), nn.ReLU(inplace)

    def forward(self, x):
        h, l = x

        return self.relu_h(h), self.relu_l(l)


class OctMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.maxpool_h = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.maxpool_l = nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):
        h, l = x

        return self.maxpool_h(h), self.maxpool_l(l)


class Error(Exception):
    """Base-class for all exceptions rased by this module."""


class InvalidOctType(Error):
    """There was a problem in the OctConv type."""
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

    def __init__(self, params, se_block_type=None, step=False):
        super(OctaveDenseBlock, self).__init__()
        print(se_block_type)
        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        # self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
        #                        kernel_size=(
        #                            params['kernel_h'], params['kernel_w']),
        #                        padding=(padding_h, padding_w),
        #                        stride=params['stride_conv'])
        # self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
        #                        kernel_size=(
        #                            params['kernel_h'], params['kernel_w']),
        #                        padding=(padding_h, padding_w),
        #                        stride=params['stride_conv'])
        # self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
        #                        kernel_size=(1, 1),
        #                        padding=(0, 0),
        #                        stride=params['stride_conv'])
        # self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        # self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        # self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        # self.prelu = nn.PReLU()


        # self.convinp = OctConv2d('first', in_channels=params['num_channels'], out_channels=params['num_filters'], kernel_size=(
        #                             params['kernel_h'], params['kernel_w']), padding=(padding_h, padding_w), 
        #                             stride=params['stride_conv'])

        self.conv1 = OctConv2d('first', in_channels=params['num_channels'], out_channels=params['num_filters'], kernel_size=(
                                    params['kernel_h'], params['kernel_w']), padding=(padding_h, padding_w), 
                                    stride=params['stride_conv'])


        self.conv2 = OctConv2d('regular', in_channels=conv1_out_size, out_channels=params['num_filters'], kernel_size=(
                                    params['kernel_h'], params['kernel_w']), padding=(padding_h, padding_w),
                                stride=params['stride_conv'])

        # self.conv3 = OctConv2d('regular', in_channels=conv2_out_size, out_channels=params['num_filters'], kernel_size=(1, 1), padding=(0, 0),
        #                         stride=params['stride_conv'])

        self.conv3 = OctConv2d('last', in_channels=conv2_out_size, out_channels=params['num_filters'], kernel_size=(
                                    params['kernel_h'], params['kernel_w']), padding=(padding_h, padding_w), 
                                    stride=params['stride_conv'])

        self.avgpool1 = nn.AvgPool2d(2)
        # self.avgpool2 = nn.AvgPool2d(2)
        # self.avgpool3 = nn.AvgPool2d(2)


        # self.batchnorm1_h = None if alpha_in == 1 else nn.BatchNorm2d(
        #     num_features=int(params['num_channels'] * (1 - alpha_in)))
        # self.batchnorm1_l = None if alpha_in == 0 else nn.BatchNorm2d(
        #     num_features=int(params['num_channels'] * alpha_in))
        alpha_in, alpha_out = 0.5, 0.5

        # self.batchnorm2_h = None if alpha_out == 1 else nn.BatchNorm2d(
        #     num_features=int(conv1_out_size * (1 - alpha_out)))
        # self.batchnorm2_l = None if alpha_out == 0 else nn.BatchNorm2d(num_features=int(conv1_out_size * alpha_out))

        # self.batchnorm3_h = None if alpha_out == 1 else nn.BatchNorm2d(
        #     num_features=int(conv2_out_size * (1 - alpha_out)))
        # self.batchnorm3_l = None if alpha_out == 0 else nn.BatchNorm2d(num_features=int(conv2_out_size * alpha_out))
        print(step)
        if step:
            self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])

            self.batchnorm2_h = nn.BatchNorm2d(num_features=int(96)) #conv1_out_size * (1 - alpha_out))+1)
            self.batchnorm2_l = nn.BatchNorm2d(num_features=int(96)) #conv1_out_size-1))

            self.batchnorm3_h = nn.BatchNorm2d(num_features=int(128)) #conv2_out_size * (1 - alpha_out))+1)
            self.batchnorm3_l = nn.BatchNorm2d(num_features=int(128)) #conv2_out_size-1))
        else:
            self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])

            self.batchnorm2_h = nn.BatchNorm2d(num_features=int(64)) #conv1_out_size * (1 - alpha_out)
            self.batchnorm2_l = nn.BatchNorm2d(num_features=int(64)) #(conv1_out_size * alpha_out))

            self.batchnorm3_h = nn.BatchNorm2d(num_features=int(96)) #conv2_out_size * (1 - alpha_out)))
            self.batchnorm3_l = nn.BatchNorm2d(num_features=int(96))  #conv2_out_size * alpha_out))

        self.prelu = nn.PReLU()
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

        # o1 = self.batchnorm1(input)
        # o2 = self.prelu(o1)
        # o3 = self.conv1(o2)
        # o4 = torch.cat((input, o3), dim=1)
        # o5 = self.batchnorm2(o4)
        # o6 = self.prelu(o5)
        # o7 = self.conv2(o6)
        # o8 = torch.cat((input, o3, o7), dim=1)
        # o9 = self.batchnorm3(o8)
        # o10 = self.prelu(o9)
        # out = self.conv3(o10)
        # return out

        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        ch = input.shape[1] // 2
        inp_h, inp_l = input[:, :ch, :, :], input[:, ch:, :, :]
        inp_ll = self.avgpool1(inp_l)

        o3_h, o3_l = self.conv1(o2)
        o4_h = torch.cat((inp_h, o3_h), dim=1)
        o4_l = torch.cat((inp_ll, o3_l), dim=1)
        o5_h = self.batchnorm2_h(o4_h)
        o5_l = self.batchnorm2_l(o4_l)
        o6_h = self.prelu_h(o5_h)
        o6_l = self.prelu_l(o5_l)
        o7_h, o7_l = self.conv2((o6_h, o6_l))

        # inp_lll = self.avgpool2(inp_ll)
        # o3_ll = self.avgpool3(o3_l)


        o8_h = torch.cat((inp_h, o3_h, o7_h), dim=1)
        o8_l = torch.cat((inp_ll, o3_l, o7_l), dim=1)

        o9_h = self.batchnorm3_h(o8_h)
        o9_l = self.batchnorm3_l(o8_l)
        o10_h = self.prelu_h(o9_h)
        o10_l = self.prelu_l(o9_l)
        out = self.conv3((o10_h, o10_l))

        return out


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

    def __init__(self, params, se_block_type=None, step=False):
        super(OctaveEncoderBlock, self).__init__(params, se_block_type=se_block_type, step=step)
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

        out_block = super(OctaveEncoderBlock, self).forward(input)
        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


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

    def __init__(self, params, se_block_type=None, step=False):
        super(OctaveDecoderBlock, self).__init__(params, se_block_type=se_block_type, step=step)
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
        if indices is not None:
            unpool = self.unpool(input, indices)
        else:
            # TODO: Implement Conv Transpose
            print("You have to use Conv Transpose")

        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(OctaveDecoderBlock, self).forward(concat)

        if self.SELayer:
            out_block = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block
