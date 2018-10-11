import torch

def create_label_mask(input_size, stride, anchors,labels,threshold):
    '''
    Create the label information tensor for a feature map
    :param input_size: 1d int tensor, (z,h,w)
    :param stride: int
    :param anchors: id float tensor, (d1,d2,...,dn)
    :param labels: float tensor, shape: (bs,N) N=5 (label,iz,ih,iw,d)
    :param threshold: (bs,
    :return:
    '''
    pass