# gradTestTrHess.py

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.Phi import Phi
import torch.nn.utils
import copy
import torch.nn as nn



doPlots = True


d = 2
m = 5
nTh = 3

net = Phi(nTh=nTh, m=m, d=d)
net.double()

# vecParams = nn.utils.convert_parameters.parameters_to_vector(net.parameters())
x = torch.randn(1,3).type(torch.double)
x.requires_grad = True
y = net(x)
v = torch.randn(x.shape).type(torch.double)

# ------------------------------------------------
# f
# nablaPhi = net.trHess(x)[0]

g = net.trHess(x)[0]

niter = 20
h0 = 0.1
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
# ------------------------------------------------
# f is the gradient wrt x  computation
# trH = net.trHess(x)[1]

input = torch.randn(1,3).type(torch.double)

fx = net.trHess(input)[1]
vecX = copy.copy(nn.utils.convert_parameters.parameters_to_vector(net.parameters()))
v = torch.randn(vecX.shape).type(torch.double)

netV = Phi(nTh=nTh, m=m, d=d).double() # make another copy for shape info

# jacobian of fx wrt x
# g = torch.autograd.grad(fx,net.w.weight, retain_graph=True, create_graph=True, allow_unused=True)
g = torch.autograd.grad(fx,net.parameters(), retain_graph=True, create_graph=True, allow_unused=True)

nn.utils.convert_parameters.vector_to_parameters(v, netV.parameters())  # structure v into the tensors
gv = 0.0
for gi, vi in zip(g, netV.parameters()):
    if gi is not None:  # if gi is None, then that means the gradient there is 0
        gv += torch.matmul(gi.view(1, -1), vi.view(-1, 1))

niter = 20
h0 = 0.1
E0 = []
E1 = []
hlist = []

for i in range(1,niter):
    h = h0**i
    hlist.append(h)

    newVec = vecX + h*v

    nn.utils.convert_parameters.vector_to_parameters(newVec, net.parameters())  # set parameters
    fxhv = net.trHess(input)[1]

    # print(newVec[0:3])
    # print(vecX[0:3])
    # print("{:.6f}  {:.6f}  {:.6e}".format(fxhv.item(), fx.item() , torch.norm(net.w.weight - torch.ones(5).type(torch.double)).item()))

    fdiff = fxhv - fx

    res0 = torch.norm(fdiff)
    E0.append( res0  )

    res1 = torch.norm(fdiff - h * gv)
    E1.append( res1  )

print(" ")
for i in range(niter-1):
    print("{:e}  {:.6e}  {:.6e}".format( hlist[i] , E0[i].item() , E1[i].item() ))


if doPlots:
    plt.plot(hlist,E0, label='E0')
    plt.plot(hlist,E1, label='E1')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()


print("\n")
# ------------------------------------------------
# f is the trace of the Hessian computation
# trH = net.trHess(x)[0]

input = torch.randn(1,3).type(torch.double)

fx = torch.sum(net.trHess(input)[0])
vecX = copy.copy(nn.utils.convert_parameters.parameters_to_vector(net.parameters()))
v = torch.randn(vecX.shape).type(torch.double)

netV = Phi(nTh=nTh, m=m, d=d).double() # make another copy for shape info

# jacobian of fx wrt x
# g = torch.autograd.grad(fx,net.w.weight, retain_graph=True, create_graph=True, allow_unused=True)
g = torch.autograd.grad(fx,net.parameters(), retain_graph=True, create_graph=True, allow_unused=True)

nn.utils.convert_parameters.vector_to_parameters(v, netV.parameters())  # structure v into the tensors
gv = 0.0
for gi, vi in zip(g, netV.parameters()):
    if gi is not None:  # if gi is None, then that means the gradient there is 0
        gv += torch.matmul(gi.view(1, -1), vi.view(-1, 1))



niter = 20
h0 = 0.1
E0 = []
E1 = []
hlist = []


for i in range(1,niter):
    h = h0**i
    hlist.append(h)

    newVec = vecX + h*v

    nn.utils.convert_parameters.vector_to_parameters(newVec, net.parameters())  # set parameters
    fxhv = torch.sum(net.trHess(input)[0])

    # print(newVec[0:3])
    # print(vecX[0:3])
    # print("{:.6f}  {:.6f}  {:.6e}".format(fxhv.item(), fx.item() , torch.norm(net.w.weight - torch.ones(5).type(torch.double)).item()))

    fdiff = fxhv - fx

    res0 = torch.norm(fdiff)
    E0.append( res0  )

    res1 = torch.norm(fdiff - h * gv)
    E1.append( res1  )

print(" ")
for i in range(niter-1):
    print("{:e}  {:.6e}  {:.6e}".format( hlist[i] , E0[i].item() , E1[i].item() ))


if doPlots:
    plt.plot(hlist,E0, label='E0')
    plt.plot(hlist,E1, label='E1')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()











