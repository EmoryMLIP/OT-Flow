# testPhiOpt.py

from src.Phi import Phi
from src.PhiHC import PhiHC
from src.OTFlowProblem import *
import torch
import time

# we know PhiHardCoded is accurate, nTh is hardcoded in there
# we want to generalize it so nTh > 2

if __name__ == '__main__':

    d = 8
    m = 16
    nTh = 2
    alph = [1.0,1.0,3.0,5.0,1.0]

    torch.manual_seed(0)
    net = Phi(nTh, m, d, alph=alph)
    net = net.to(torch.double)

    torch.manual_seed(0)
    netLoop = PhiHC(nTh, m,d,alph=alph)
    netLoop = netLoop.to(torch.double)

    nex = 10000
    x = torch.randn(nex,d+1).to(torch.double)

    end = time.time()
    y = net(x)
    print("time: ", time.time()-end)

    end = time.time()
    yLoop = netLoop(x)
    print("time: ", time.time()-end)


    print("Phi       err: ", torch.norm(y-yLoop).item())

    end = time.time()
    y1,y2 = net.trHess(x)
    print("time: ", time.time()-end)
    end = time.time()
    yLoop1,yLoop2 = netLoop.trHess(x)
    print("time: ", time.time()-end)
    print("grad      err: ", torch.norm(y1 - yLoop1).item())
    print("traceHess err: ", torch.norm(y2 - yLoop2).item())


