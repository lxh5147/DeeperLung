import torch
from torch import nn

from .nodule_constant import POSITIVE_LABEL, NEGATIVE_LABEL, FIXED_NEGATIVE_LABEL


def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels, idcs


def select_value(output, labels, value):
    idcs = labels[:, 0] == value
    select_output = output[idcs]
    select_labels = labels[idcs]
    return select_output, select_labels, idcs


class Loss(nn.Module):
    def __init__(self, num_hard_neg_per_patch=0, cls_weight=(0.5, 0.5, 0.2), reg_weight=1.0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.classify_loss = nn.BCELoss()
        self.classify_loss = nn.BCEWithLogitsLoss()
        self.regress_loss = nn.SmoothL1Loss()

        self.num_hard_neg_per_patch = num_hard_neg_per_patch
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, output, labels, train=True):

        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_output, pos_labels, pos_indices = select_value(output, labels, POSITIVE_LABEL)

        neg_output, neg_labels, neg_indices = select_value(output, labels, NEGATIVE_LABEL)
        neg_output = neg_output[:, 0]
        neg_labels = neg_labels[:, 0]

        # In train and val phase, we both choose top k negatives.
        if self.num_hard_neg_per_patch > 0:
            neg_output, neg_labels, neg_topk_idcs = hard_mining(neg_output, neg_labels,
                                                                self.num_hard_neg_per_patch * batch_size)

        neg_prob = self.sigmoid(neg_output)

        fixed_neg_output, fixed_neg_labels, fixed_neg_idcs = select_value(output, labels, FIXED_NEGATIVE_LABEL)

        fixed_neg_output = fixed_neg_output[:, 0]
        fixed_neg_labels = fixed_neg_labels[:, 0]
        fixed_neg_prob = self.sigmoid(fixed_neg_output)

        pos_loss = 0
        pos_correct = 0
        pos_total = 0
        neg_loss = 0
        neg_correct = 0
        neg_total = 0
        fixed_neg_loss = 0
        fixed_neg_correct = 0
        fixed_neg_total = 0

        regress_losses = [0] * 4


        if len(pos_output) > 0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]

            pos_loss = self.classify_loss(pos_output[:, 0], pos_labels[:, 0])
            pos_correct = (pos_prob >= 0.5).sum()
            pos_total = len(pos_prob)

        if len(neg_output) > 0:
            neg_loss = self.classify_loss(neg_output, neg_labels - NEGATIVE_LABEL)
            neg_correct = (neg_prob.data < 0.5).sum()
            neg_total = len(neg_prob)

        if len(fixed_neg_output) > 0:
            fixed_neg_loss = self.classify_loss(fixed_neg_output, fixed_neg_labels - FIXED_NEGATIVE_LABEL)
            fixed_neg_correct = (fixed_neg_prob.data < 0.5).sum()
            fixed_neg_total = len(fixed_neg_prob)

        pos_loss_data = pos_loss.item()
        neg_loss_data = neg_loss.item()
        fixed_neg_loss_data = fixed_neg_loss.item()

        loss = self.cls_weight[0] * pos_loss + self.cls_weight[1] * neg_loss + self.cls_weight[2] * fixed_neg_loss
        for regress_loss in regress_losses:
            loss += self.reg_weight * regress_loss

        # return tensor
        return [loss, pos_loss_data, neg_loss_data, fixed_neg_loss_data] + regress_losses_data + \
               [pos_correct, pos_total, neg_correct, neg_total, fixed_neg_correct, fixed_neg_total]


class FocalLoss(nn.Module):
    r"""
        This criterion is a implementation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
        for multi class

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        for binary class
            # focus on hard case
            Loss(x, pos) = - \alpha (1-sigmoid(x))^gamma \ log(sigmoid(x))
            Loss(x, neg) = - \alpha (sigmoid(x))^gamma \ log(1-sigmoid(x))

        The losses are averaged across observations for each minibatch.
        For now, only support binary class.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, num_hard=0, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_hard = num_hard
        if alpha is None:
            self.alpha = nn.Parameter(torch.ones(1, ))
        else:
            self.alpha = nn.Parameter(alpha)
        self.gamma = gamma
        self.size_average = size_average
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, output, labels, train=True):
        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs = labels[:, 0] == POSITIVE_LABEL
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] == NEGATIVE_LABEL
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels, _ = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        if len(pos_output) > 0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]

            pos_loss = -self.alpha * torch.pow(1 - pos_prob, self.gamma) * (pos_prob + 0.0001).log()
            neg_loss = -(1 - self.alpha) * torch.pow(neg_prob, self.gamma) * (1 - neg_prob + 0.0001).log()

            if self.size_average:
                classify_loss = (pos_loss.sum() + neg_loss.sum()) / pos_loss.size()[0]
            else:
                classify_loss = pos_loss.sum() + neg_loss.sum()

            pos_correct = (pos_prob.data >= 0.7).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            regress_losses_data = [0, 0, 0, 0]
            neg_loss = -(1 - self.alpha) * torch.pow(neg_prob, self.gamma) * (1 - neg_prob).log()

            classify_loss = neg_loss.sum()

            pos_correct = 0
            pos_total = 0

        classify_loss_data = classify_loss.item()

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.3).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]
