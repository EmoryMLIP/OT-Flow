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
        # compute vector-Jacobain Product using AD with the rademacher vector
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
    _ = compareTrace(domainMini, 50) # dry-run
    resTimeBSDS , traceErrorBSDS , exactTimingBSDS = compareTrace(domainBSDS, 63)
    _ = compareTrace(domainMini, 50) # dry-run
    resTimeMini , traceErrorMini, exactTimingMini = compareTrace(domainMini, 43) 
    _ = compareTrace(domainMini, 50) # dry-run
    resTimeMNIST, traceErrorMNIST, exactTimingMNIST = compareTrace(domainMNIST, 784)

    for i in range(nRepeats-1):
        _ = compareTrace(domainMini, 50) # dry-run
        a , b, c = compareTrace(domainBSDS, 63)
        resTimeBSDS     += a
        traceErrorBSDS  += b
        exactTimingBSDS += c
        _ = compareTrace(domainMini, 50) # dry-run
        a , b, c = compareTrace(domainMini, 43)
        resTimeMini     += a
        traceErrorMini  += b
        exactTimingMini += c
        _ = compareTrace(domainMini, 50) # dry-run
        a , b, c = compareTrace(domainMNIST, 784)
        resTimeMNIST    += a
        traceErrorMNIST += b
        exactTimingMNIST+= c

    # average and convert to list
    resTimeMini       = (resTimeMini/nRepeats).view(-1).tolist()
    traceErrorMini    = (traceErrorMini/nRepeats).view(-1).tolist()
    exactTimingMini  /= nRepeats
    resTimeBSDS       = (resTimeBSDS/nRepeats).view(-1).tolist()
    traceErrorBSDS    = (traceErrorBSDS/nRepeats).view(-1).tolist()
    exactTimingBSDS  /= nRepeats
    resTimeMNIST      = (resTimeMNIST/nRepeats).view(-1).tolist()
    traceErrorMNIST   = (traceErrorMNIST/nRepeats).view(-1).tolist()
    exactTimingMNIST /= nRepeats



    # print out the values
    torch.set_printoptions(precision=10)
    print("values to plot")
    print("-----------------")
    print("Miniboone")
    print("AD timings:   ", resTimeMini)
    print("AD Error:     ", traceErrorMini)
    print("Exact timing: ", exactTimingMini)
    print("BSDS")
    print("AD timings:   ", resTimeBSDS)
    print("AD Error:     ", traceErrorBSDS)
    print("Exact timing: ", exactTimingBSDS)
    print("MNIST")
    print("AD timings:   ", resTimeMNIST)
    print("AD Error:     ", traceErrorMNIST)
    print("Exact timing: ", exactTimingMNIST)


    lTimeExact = [exactTimingMini, exactTimingBSDS, exactTimingMNIST]
    plotTraceCompare(domainMini    , domainBSDS    , domainMNIST,
                     resTimeMini   , resTimeBSDS   , resTimeMNIST,
                     traceErrorMini, traceErrorBSDS, traceErrorMNIST,
                     lTimeExact, 'image/traceComparison/')


