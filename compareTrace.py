# compareTrace.py
# compare the exact trace in Phi with the hutchinsons estimator using atomatic differentiation

import math
from src.OTFlowProblem import *

gpu = 0
if not torch.cuda.is_available():
    print("No gpu found. If you wish to run on a CPU, remove the cuda specific lines, then run again.")
    exit(1)


device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------------------------------------------------------------
# compare timings with AD
# ----------------------------------------------------------------------------------------------------------------------

def compareTrace(domain,d, seed=0):
    """
    domain: list of integers specificying the number of hutchinson vectors to use
    d:      dimensionality of the problem
    :param domain:
    :return:
    """

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.manual_seed(seed) # for reproducibility

    # set up model
    m    = 64
    alph = [1.0,1.0,1.0,1.0,1.0]
    nTh  = 2
    net = Phi(nTh=nTh, m=m, d=d, alph=alph)
    net = net.to(device)

    n_samples = 512
    x = torch.randn(n_samples, d+1).to(device)
    x.requires_grad = True


    # dry-run / warm-up
    start.record()
    a = torch.Tensor(2000,3000)
    b = torch.Tensor(3000,4000)
    c = torch.mm(a,b)
    end.record()
    torch.cuda.synchronize()
    _ = start.elapsed_time(end) / 1000.0 # convert to seconds 


    # ---------------------------------------------
    # time the exact trace
    # ---------------------------------------------
    
    start.record()
    grad, exact_trace = net.trHess(x)
    end.record()
    torch.cuda.synchronize() 
    exact_time = start.elapsed_time(end) / 1000.0 # convert to seconds 
    print("Exact Trace Computation time= {:9.6f}".format(exact_time))

    # ---------------------------------------------
    # time hutchinson's estimator using AD
    # compute an estimate for each value in domain
    # aka domain=[1,10,20] will run an estiamte with 1 hutch vector, one with 10 hutch vectors, and one with 20 hutch vectors
    # ---------------------------------------------

    # where to hold results
    resTime = torch.zeros(1,len(domain))
    resErr  = torch.zeros(1,len(domain))

    for iDomain, num_hutchinsons in enumerate(domain):
        torch.manual_seed(seed+1)
        trace_acc = torch.zeros(n_samples).to(device) # accumulated trace


        # create the num_hutchinsons rademacher vectors...these "vectors" are each stored as a matrix
        # we have num_hutchinsons of them, so that makes a tensor called rad
        # compute vector-Jacobian Product using AD with the rademacher vector


        start.record()
        rad = (1 / math.sqrt(num_hutchinsons)) * ((torch.rand(n_samples, d+1, num_hutchinsons,device=device) < 0.5).float() * 2 - 1)
        rad[:,d,:] = 0 # set time position to 0, leave space values as rademacher
        # rad = rad.to(device)
        for i in range(num_hutchinsons):
            e = rad[:,:,i] # the "random vector"
            grad = net.trHess(x, justGrad=True)
            trace_est = torch.autograd.grad(outputs=grad, inputs=x, create_graph=False,retain_graph=False, grad_outputs=e)[0]
            trace_est = trace_est * e
            trace_est = trace_est.view(grad.shape[0], -1).sum(dim=1)
            trace_acc += trace_est
        end.record()
        torch.cuda.synchronize()
        ad_time = start.elapsed_time(end) / 1000.0 # convert to seconds
        
        trace_error = torch.norm(exact_trace-trace_acc)/torch.norm(exact_trace) # compute error
        print("{:4d} hutchinson vectors.  time= {:9.6f} , rel. error = {:9.7f}".format(num_hutchinsons, ad_time, trace_error ))
        resTime[0, iDomain] = ad_time
        resErr[0, iDomain]  = trace_error

    # return timings nad errors for plotting/analysis
    return resTime, resErr, exact_time

if __name__ == '__main__':

    from src.plotTraceComparison import *

    domainMini  = [1, 10, 20, 30, 43]
    domainBSDS  = [1, 10, 20, 30, 40, 50, 63]
    domainMNIST = [1, 100, 200, 300, 400, 500, 600, 700, 784]
    
    nRepeats = 2 # average over 2 runs. For publication figure, we set this to 20

    # arrays to hold all the results...in case we want to use error bounds
    resTimeBSDSArray            = torch.zeros(nRepeats, len(domainBSDS))
    traceErrorBSDSArray         = torch.zeros(nRepeats, len(domainBSDS))
    exactTimingBSDSArray        = torch.zeros(nRepeats, 1)
    resTimeMiniArray            = torch.zeros(nRepeats, len(domainMini))
    traceErrorMiniArray         = torch.zeros(nRepeats, len(domainMini))
    exactTimingMiniArray        = torch.zeros(nRepeats, 1)
    resTimeMNISTArray           = torch.zeros(nRepeats, len(domainMNIST))
    traceErrorMNISTArray        = torch.zeros(nRepeats, len(domainMNIST))
    exactTimingMNISTArray       = torch.zeros(nRepeats, 1)

    for i in range(nRepeats):
        print('\n\n ITER ', i)
        _ = compareTrace(domainMini, 50) # dry-run
        a , b, c = compareTrace(domainBSDS, 63, seed=i);
        resTimeBSDSArray[i,:] = a
        traceErrorBSDSArray[i,:] = b
        exactTimingBSDSArray[i] = c
        _ = compareTrace(domainMini, 50) # dry-run
        a , b, c = compareTrace(domainMini, 43, seed=i)
        resTimeMiniArray[i, :] = a
        traceErrorMiniArray[i, :] = b
        exactTimingMiniArray[i] = c
        _ = compareTrace(domainMini, 50) # dry-run
        a , b, c = compareTrace(domainMNIST, 784, seed=i)
        resTimeMNISTArray[i, :] = a
        traceErrorMNISTArray[i, :] = b
        exactTimingMNISTArray[i] = c


    lTimeExact = [ exactTimingMiniArray  , exactTimingBSDSArray , exactTimingMNISTArray]
    plotTraceCompare(domainMini    , domainBSDS    , domainMNIST,
                     resTimeMiniArray   , resTimeBSDSArray   , resTimeMNISTArray,
                     traceErrorMiniArray, traceErrorBSDSArray, traceErrorMNISTArray,
                     lTimeExact, 'image/traceComparison/')


