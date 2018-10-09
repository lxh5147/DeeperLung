import numpy as np
import torch

class GetPBB(object):
    def __init__(self, stride, anchors):
        '''
        Initialize a callable object to get the bounding boxes from the output of the network.
        All the locations and sizes are relative to the input grid.
        :param stride: int, down sample rate of the input
        :param anchors: tensor, 1D float, the size of the anchors
        '''
        self.stride = stride
        self.anchors = anchors

    def __call__(self, output, thresh=-3.):
        '''
        Get the bounding box from the output from the network.
        :param output: tensor, output of the network, batch_size, z,h,w,a,N=5 (p,z,h,w,d), where p is logits
        :param thresh: float, if the probability is greater than this threshold, it is a nodule candidate
        :param is_mask: bool, if true the mask of the nodule candidate
        :return: float tensor with shape M,N, where M is the number of nodules with p> thresh,
            and the mask ( calulated by np.where )
        '''
        stride = self.stride
        anchors = self.anchors

        output_size = output.size()

        assert len(output_size) > 5
        assert output_size[-1] == 5 # p dz dh dw dd
        assert output_size[-2] == len(anchors)

        offset = (float(stride) - 1) / 2
        # output is the feature map
        oz = np.arange(offset, offset + stride * (output_size[-5] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[-4] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[-3] - 1) + 1, stride)
        # to shape b z h w a N, and create a copy of output
        output_copy = output.reshape( (-1,) + output_size[-5:])
        # to shape batch_size, z,h,w,a
        oz = oz.reshape((1, -1, 1, 1, 1))
        oh = oh.reshape((1, 1, -1, 1, 1))
        ow = ow.reshape((1, 1, 1, -1, 1))
        anchors = anchors.reshape((1, 1, 1, 1, -1))
        # the output from delta to absolute position in crop
        output_copy[:, :, :, :, :, 1] = oz + output_copy[:, :, :, :, :, 1] * anchors
        output_copy[:, :, :, :, :, 2] = oh + output_copy[:, :, :, :, :, 2] * anchors
        output_copy[:, :, :, :, :, 3] = ow + output_copy[:, :, :, :, :, 3] * anchors
        output_copy[:, :, :, :, :, 4] = output_copy[:, :, :, :, :, 4].exp() * anchors
        output_copy = output_copy.view(output_size)
        # shape: batch_size, z,h,w,a
        mask = output_copy[..., 0] > thresh
        # shape: #elements above threshold,N
        output_copy = output_copy[mask]
        # shape: #elements above threshold, #dimensions
        mask = mask.nonzero()
        return output_copy, mask
