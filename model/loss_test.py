import unittest
from .loss import select_value
import torch
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_select_value(self):
        output = torch.from_numpy(np.asarray([[2,3],[0,4]], dtype=np.float32))
        labels = torch.from_numpy(np.asarray([[2,3.2],[0,4.2]], dtype=np.float32))
        select_output, select_labels, idcs = select_value(output,labels,2)
        self.assertEqual(select_output.data, )
if __name__ == '__main__':
    unittest.main()
