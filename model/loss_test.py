import unittest
from .loss import select_value
import torch
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_select_value(self):
        output = torch.from_numpy(np.asarray([[2,3],[0,4]], dtype=np.float32))
        labels = torch.from_numpy(np.asarray([[2,3.2],[0,4.2]], dtype=np.float32))
        select_output, select_labels, idcs = select_value(output,labels,2)
        self.assertTrue((select_output.numpy()==np.asarray([[2,3]])).all(),'select output')
        np.testing.assert_almost_equal(select_labels.numpy(), np.asarray([[2, 3.2]]))
    def test(self):
        x = 2.
        t = torch.ones((2,))
        y=x+t
        print(y)
        t.size()[0]

