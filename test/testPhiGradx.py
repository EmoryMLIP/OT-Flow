# testPhiGradx.py
# test the grad wrt x returned by trHess when nTh > 2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.Phi import *
import torch.nn.utils

doPlots = True

d = 2
m = 5
nTh = 4

net = Phi(nTh=nTh, m=m, d=d)
net.double()

# vecParams = nn.utils.convert_parameters.parameters_to_vector(net.parameters())
x = torch.randn(1,3).type(torch.double)
# AD grad
x.requires_grad = True
y = net(x)

v = torch.randn(x.shape).type(torch.double)

# ------------------------------------------------
# f
# nablaPhi = net.trHess(x)[0]

g = net.trHess(x)[0]


niter = 20
h0 = 0.5
E0 = []
E1 = []
hlist = []


for i in range(niter):
    h = h0**i
    hlist.append(h)
    E0.append( torch.norm(net( x + h * v ) - net(x))  )
    E1.append( torch.norm(net( x + h * v ) - net(x) - h * torch.matmul(g , v.t()))   )

for i in range(niter):
    print("{:f} {:.6e} {:.6e}".format( hlist[i] , E0[i].item() , E1[i].item() ))

if doPlots:
    plt.plot(hlist,E0, label='E0')
    plt.plot(hlist,E1, label='E1')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()



print("\n")