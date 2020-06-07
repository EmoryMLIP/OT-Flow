# mmd.py
# Maximum Mean Discrepancy

import torch
import numpy as np

# from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd


def mmd(x,y, indepth=False, alph=1.0):
    """
        from Li et al. Generative Moment Matching Networks 2015

    Gaussian kernel

    :param x: numpy matrix of size (nex, :)
    :param y: numpy matrix of size (nex,:)
    :return: MMD(x,y)
    """

    # convert to numpy
    if type(x) is torch.Tensor:
        x = x.numpy()
    if type(y) is torch.Tensor:
        y = y.numpy()

    if max(x.size,y.size) > 20000:
        indepth = True



    # there's a quick method, that uses a lot of memory, that can be run on pointclouds of a few thousand samples
    # and there's a long and slow way that can be run on pointclouds with 10^5 samples
    if not indepth:
        # make torch tensor
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(torch.float32)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y).to(torch.float32)
        mmdObj = MMDStatistic(x.shape[0],y.shape[0])
        return mmdObj( x , y , [alph] ).item() # just use alpha = 1.0

    else:
        # lots of examples, do a long approach
        # very slow
        # kernel  = exp(  1/(2*sig) * || x - xj ||^2    )


        # sig = 0.5
        # alpha = -1.0 / (2*sig)
        alpha = - alph

        xx  = 0.0
        yy  = 0.0
        xy  = 0.0
        N = x.shape[0]
        M = y.shape[0]

        NsqrTerm  = 1/N**2
        MsqrTerm  = 1/M**2
        crossTerm = -2/(N*M)

        for i in range(N):
            xi = x[i,:]
            diff = xi - x
            power = alpha * np.linalg.norm(diff, ord=2, axis=1, keepdims=True)**2 # nex-by-1
            xx += np.exp(power).sum()

            diff = xi - y
            power = alpha * np.linalg.norm(diff, ord=2, axis=1, keepdims=True) ** 2  # nex-by-1
            xy += np.exp(power).sum()

        for i in range(M):
            yi = y[i,:]
            diff = yi - y
            power = alpha * np.linalg.norm(diff, ord=2, axis=1, keepdims=True)**2 # nex-by-1
            yy += np.exp(power).sum()

        return NsqrTerm*xx + crossTerm*xy + MsqrTerm*yy




if __name__ == "__main__":

    for N in [2000,20000]:
        M = N-9
        d = 10
        x = 50.0 + np.random.rand(N,d)
        y = np.random.randn(M,d)
        ret = mmd(x,y)
        print('mmd: {:.3e}'.format(ret))
