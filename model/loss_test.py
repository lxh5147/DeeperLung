import unittest

import numpy as np
import torch

from .loss import select_value, Loss, FocalLoss


class MyTestCase(unittest.TestCase):
    def test_select_value(self):
        output = torch.from_numpy(np.asarray([[2,3],[0,4]], dtype=np.float32))
        labels = torch.from_numpy(np.asarray([[2,3.2],[0,4.2]], dtype=np.float32))
        select_output, select_labels, idcs = select_value(output,labels,2)
        self.assertTrue((select_output.numpy()==np.asarray([[2,3]])).all(),'select output')
        np.testing.assert_almost_equal(select_labels.numpy(), np.asarray([[2, 3.2]]))
    def test_loss(self):
        loss = Loss()
        batch_size = 2
        z = 4
        w = 4
        h = 4
        N = 5
        output = torch.ones((batch_size, z, h, w, 3, N))
        labels = torch.ones((batch_size, z, h, w, 3, N))
        losses = loss(output,labels)
        # [tensor(0.1566), tensor(0.3133), 0, 0, tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(384), 384, 0, 0, 0, 0]
        self.assertEqual(2*losses[0].item(), losses[1].item(),'simple check')

    def test_focal_loss(self):
        loss = FocalLoss()
        batch_size = 2
        z = 4
        w = 4
        h = 4
        N = 5
        output = torch.ones((batch_size, z, h, w, 3, N))
        labels = torch.ones((batch_size, z, h, w, 3, N))
        losses = loss(output, labels)
        # [tensor(0.0226, grad_fn=<ThAddBackward>), tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(384), 384, tensor(0), 0]
        self.assertEqual( losses[1].item(), 0, 'simple check')
