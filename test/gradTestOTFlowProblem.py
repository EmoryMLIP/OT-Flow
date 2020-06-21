# testOTFlowProblem.py
#

# gradient check of OTFlowProblem
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.Phi import *
from src.OTFlowProblem import *
import torch.nn.utils


doPlots = True

d = 5
m = 16

net = Phi(nTh=2, m=m, d=d)
net.double()

# vecParams = nn.utils.convert_parameters.parameters_to_vector(net.parameters())
x = torch.randn(1,d+1).type(torch.double)
# net(x)

v = torch.randn(x.shape).type(torch.double)
# ------------------------------------------------
# f is the full OTFlowProblem
# OTFlowProblem(x, Phi, tspan , nt, stepper="rk1", alph =[1.0,1.0,1.0,1.0,1.0] )

input = torch.randn(1,d+1).type(torch.double)

fx = torch.sum(OTFlowProblem(input[:,0:d], net, [0.0, 1.0] , nt=2, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0] )[0])
vecX = copy.copy(nn.utils.convert_parameters.parameters_to_vector(net.parameters()))
v = torch.randn(vecX.shape).type(torch.double)

netV = Phi(nTh=2, m=m, d=d).double() # make another copy for shape info

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
    fxhv = torch.sum(OTFlowProblem(input[:,0:d], net, [0.0, 1.0] , nt=2, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0] )[0])

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














# CHECK JUST ONE PARAMETER TENSOR

# d = 5
# m = 16
#
# net = Phi(nTh=2, m=m, d=d)
# net.double()


# ------------------------------------------------
# f is the full OTFlowProblem
# OTFlowProblem(x, Phi, tspan , nt, stepper="rk1", alph =[1.0,1.0,1.0,1.0,1.0] )

input = torch.randn(1,d+1).type(torch.double)

fx = torch.sum(OTFlowProblem(input[:,0:d], net, [0.0, 1.0] , nt=2, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0] )[0])
x = net.N.layers[0].weight.data
v = torch.randn(net.N.layers[0].weight.data.shape).type(torch.double)

# netV = Phi(nTh=2, m=m, d=d).double() # make another copy for shape info

# jacobian of fx wrt x
# g = torch.autograd.grad(fx,net.w.weight, retain_graph=True, create_graph=True, allow_unused=True)
g = torch.autograd.grad(fx,net.N.layers[0].weight, retain_graph=True, create_graph=True, allow_unused=True)[0]

# nn.utils.convert_parameters.vector_to_parameters(v, netV.parameters())  # structure v into the tensors
# gv = 0.0
# for gi, vi in zip(g, netV.parameters()):
#     if gi is not None:  # if gi is None, then that means the gradient there is 0
#         gv += torch.matmul(gi.view(1, -1), vi.view(-1, 1))

gv = torch.matmul(g.view(1, -1), v.view(-1, 1))


niter = 20
h0 = 0.1
E0 = []
E1 = []
hlist = []


for i in range(1,niter):
    h = h0**i
    hlist.append(h)

    net.N.layers[0].weight.data = x + h*v

    fxhv = torch.sum(OTFlowProblem(input[:,0:d], net, [0.0, 1.0] , nt=2, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0] )[0])

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












