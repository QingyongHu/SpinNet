import numpy as np
import torch
import torch.nn as nn


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    """
    return torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """

    diffs = all_diffs(a, b)
    if metric == 'sqeuclidean':
        return torch.sum(diffs ** 2, dim=-1)
    elif metric == 'euclidean':
        return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
    elif metric == 'cityblock':
        return torch.sum(torch.abs(diffs), dim=-1)
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))


class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', safe_radius=0.25):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.metric = metric
        self.safe_radius = safe_radius

    def forward(self, anchor, positive):
        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        dist = cdist(anchor, positive, metric=self.metric)
        return self.calculate_loss(dist, pids)

    def calculate_loss(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.

        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """
        # generate the mask that mask[i, j] reprensent whether i th and j th are from the same identity.
        # torch.equal is to check whether two tensors have the same size and elements
        # torch.eq is to computes element-wise equality
        same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))

        # dists * same_identity_mask get the distance of each valid anchor-positive pair.
        furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
        closest_negative, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=1)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]
        loss = torch.max(furthest_positive - self.pos_margin, torch.zeros_like(diff)) + torch.max(
            self.neg_margin - closest_negative, torch.zeros_like(diff))

        return torch.mean(loss), accuracy
