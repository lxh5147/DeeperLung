import numpy as np


class GetPBB(object):
    def __init__(self, stride, anchors):
        '''
        Initialize a callable object to get the bounding boxes from the output of the network.
        All the locations and sizes are relative to the input grid.
        :param stride: int, down sample rate of the input
        :param anchors: numpy ndarray, 1D float, the size of the anchors
        '''
        self.stride = stride
        self.anchors = anchors

    def __call__(self, output, thresh=-3.):
        '''
        Get the bounding box from the output from the network.
        :param output: ndarray, output of the network, batch_size, z,h,w,a,N=5 (p,z,h,w,d), where p is logits
        :param thresh: float, if the probability is greater than this threshold, it is a nodule candidate
        :param is_mask: bool, if true the mask of the nodule candidate
        :return: float ndarray with shape M,N, where M is the number of nodules with p> thresh,
            and the mask ( calulated by np.where )
        '''
        stride = self.stride
        anchors = self.anchors

        output = np.copy(output)
        output_size = output.shape

        assert len(output_size) > 5
        assert output_size[-1] == 5 # p dz dh dw dd
        assert output_size[-2] == len(anchors)

        offset = (float(stride) - 1) / 2
        # output is the feature map
        oz = np.arange(offset, offset + stride * (output_size[-5] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[-4] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[-3] - 1) + 1, stride)
        # to shape b z h w a N
        output = np.reshape(output, (-1,) + output_size[-5:])
        # to shape batch_size, z,h,w,a
        oz = oz.reshape((1, -1, 1, 1, 1))
        oh = oh.reshape((1, 1, -1, 1, 1))
        ow = ow.reshape((1, 1, 1, -1, 1))
        anchors = anchors.reshape((1, 1, 1, 1, -1))
        # the output from delta to absolute position in crop
        output[:, :, :, :, :, 1] = oz + output[:, :, :, :, :, 1] * anchors
        output[:, :, :, :, :, 2] = oh + output[:, :, :, :, :, 2] * anchors
        output[:, :, :, :, :, 3] = ow + output[:, :, :, :, :, 3] * anchors
        output[:, :, :, :, :, 4] = np.exp(output[:, :, :, :, :, 4]) * anchors
        output = output.reshape(output_size)
        # mask is tuple of arrays, one dimension corresponds to one array
        mask = np.where(output[..., 0] > thresh)
        # shape: M,N, M is the number of predictions with prob > thresh
        output = output[mask]
        return output, mask
