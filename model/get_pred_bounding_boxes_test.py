import unittest
from .get_pred_bounding_boxes import GetPBB
import numpy as np
import torch
class MyTestCase(unittest.TestCase):
    def test_GetPBB(self):
        stride=2
        anchors=torch.from_numpy(np.asarray([3,6,8], dtype=np.float32))
        get_pbb = GetPBB(stride,anchors)
        batch_size=2
        z=4
        w=4
        h=4
        N = 5
        # original input stride * (z,w,h)
        # one naive test case
        output = torch.ones((batch_size,z,h,w,len(anchors),N))
        thresh = .5
        nodules, mask = get_pbb(output, thresh)
        self.assertEqual(nodules.size(), (batch_size*z*w*h*len(anchors),N),'nodules shape')
        self.assertEqual(mask.size(),(batch_size*z*w*h*len(anchors),5),'mask shape') # each dimension has an array


if __name__ == '__main__':
    unittest.main()
