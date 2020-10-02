"""FastSurferCNN architecture"""
import numpy as np
import torch
import torch.nn as nn

from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se

class FastSurferCNN(nn.Module):
    """
    A PyTorch implementation of QuickNAT

    """
    def __init__(self, params):
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
        super(FastSurferCNN, self).__init__()

        self.encode1 = sm.FSInputBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = 64
        self.encode2 = sm.FSEncoderBlock(params, se_block_type=params['se_block'])
        self.encode3 = sm.FSEncoderBlock(params, se_block_type=params['se_block'])
        self.encode4 = sm.FSEncoderBlock(params, se_block_type=params['se_block'])
        self.bottleneck = sm.FSDenseBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = 64
        self.decode1 = sm.FSDecoderBlock(params, se_block_type=params['se_block'])
        self.decode2 = sm.FSDecoderBlock(params, se_block_type=params['se_block'])
        self.decode3 = sm.FSDecoderBlock(params, se_block_type=params['se_block'])
        self.decode4 = sm.FSDecoderBlock(params, se_block_type=params['se_block'])
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)


    def forward(self, input):
            """

            :param input: X
            :return: probabiliy map
            """
            e1, out1, ind1 = self.encode1.forward(input)
            e2, out2, ind2 = self.encode2.forward(e1)
            e3, out3, ind3 = self.encode3.forward(e2)
            e4, out4, ind4 = self.encode4.forward(e3)

            bn = self.bottleneck.forward(e4)

            d4 = self.decode4.forward(bn, out4, ind4)
            d3 = self.decode1.forward(d4, out3, ind3)
            d2 = self.decode2.forward(d3, out2, ind2)
            d1 = self.decode3.forward(d2, out1, ind1)
            prob = self.classifier.forward(d1)

            return prob

    def enable_test_dropout(self):
        """
        Enables test time drop out for uncertainity
        :return:
        """
        attr_dict = self.__dict__['_modules']
        for i in range(1, 5):
            encode_block, decode_block = attr_dict['encode' + str(i)], attr_dict['decode' + str(i)]
            encode_block.drop_out = encode_block.drop_out.apply(nn.Module.train)
            decode_block.drop_out = decode_block.drop_out.apply(nn.Module.train)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

    def predict(self, X, device=0, enable_dropout=False, out_prob=False):
        """
        Predicts the outout after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()

        if type(X) is np.ndarray:
            X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor).cuda(device, non_blocking=True)
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)

        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        if out_prob:
            return out
        else:
            max_val, idx = torch.max(out, 1)
            idx = idx.data.cpu().numpy()
            prediction = np.squeeze(idx)
            del X, out, idx, max_val
            return prediction
