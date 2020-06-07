# PhiHC.py
# Phi Hardcoded version
# hard coded nTh = 2
import torch
import torch.nn as nn
import copy
import math

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.log( torch.exp(x) + torch.exp(-x) )

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: 2, hard-coded number of ResNet layers
        """
        super().__init__()

        nTh = 2
        self.opening   = nn.Linear(d+1 , m , bias=True)
        self.layer1 = nn.Linear(m,m, bias=True)
        self.act = antiderivTanh
        self.h = 1.0
        self.d = d
        self.m = m

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.opening(x))
        x = x + self.h * self.act(self.layer1(x))

        return x



class PhiHC(nn.Module):
    def __init__(self, nTh, m, d, r=10, alph=[1.0]*5):
        """
            neural network approximating Phi
            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:  int, number of resNet layers, hardcoded as 2
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d
        self.alph = alph

        r = min(r,d+1) # if number of dimensions is smaller than default r, use that


        self.A  = nn.Parameter(torch.zeros(r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N  = ResNN(d,m, nTh=nTh)

        # set start values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, x):

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A)

        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)

    def trHess(self, x , justGrad=False):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi)
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        N    = self.N
        m    = N.opening.weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1] - 1
        symA = torch.matmul(self.A.t(), self.A)

        # Forward of ResNet N
        opening  = N.opening(x) # K_0 * S + b_0
        u0       = N.act(opening)
        tanhopen = torch.tanh(opening)
        out1     = N.layer1(u0).t()

        # compute gradient
        z1   = self.w.weight.t() + N.h * torch.mm( N.layer1.weight.t() , torch.tanh(out1) )
        z0   = torch.mm( N.opening.weight.t() , tanhopen.t() * z1 )
        grad = z0 + torch.mm(symA, x.t() ) + self.c.weight.t()
        if justGrad:
            return grad.t()

        Kopen = N.opening.weight[:,0:d]  # Kopen = torch.mm( N.opening.weight, E  )
        trH1  = torch.sum((derivTanh(opening.t())*z1).view(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1))

        Jac  = Kopen.unsqueeze(2) * tanhopen.t().unsqueeze(1)
        # Jac is shape m by d by nex

        Jac  = torch.mm(N.layer1.weight , Jac.view(m,-1) ).view(m,-1,nex)
        trH2 = torch.sum(  (derivTanh(out1) * self.w.weight.t()).view(m,-1,nex)  *  torch.pow(Jac,2) ,  dim=(0, 1) )

        return grad.t(), trH1 + trH2 + torch.trace(symA[0:d,0:d])



if __name__ == "__main__":

    import time

    # test case
    d = 2
    m = 16

    net = PhiHC(nTh=2, m=m, d=d)
    net.N.opening.weight.data = 0.1 + 0.0 * net.N.opening.weight.data
    net.N.opening.bias.data   = 0.2 + 0.0 * net.N.opening.bias.data
    net.N.layer1.weight.data  = 0.3 + 0.0 * net.N.layer1.weight.data
    net.N.layer1.bias.data    = 0.4 + 0.0 * net.N.layer1.bias.data

    # number of samples-by-(d+1)
    x = torch.Tensor([[1.0 ,4.0 , 0.5],[2.0,5.0,0.6],[3.0,6.0,0.7],[0.0,0.0,0.0]])
    y = net(x)
    print(y)


    # test timings
    d = 400
    m = 32
    nex = 1000

    net = PhiHC(nTh=2, m=m, d=d)
    x = torch.randn(nex,d+1)
    y = net(x)

    end = time.time()
    g,h = net.trHess(x)
    print('traceHess takes ', time.time()-end)

    end = time.time()
    g = net.trHess(x, justGrad=True)
    print('JustGrad takes  ', time.time()-end)



