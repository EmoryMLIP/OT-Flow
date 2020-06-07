# MeanFieldGame.py
import math
import torch
from torch.nn.functional import pad
from src.Phi import *


def vec(x):
    """vectorize torch tensor x"""
    return x.view(-1,1)

def MeanFieldGame(x, Phi, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0] ):
    """

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of lenght 5, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 4, 0, 0), value=0)

    tk = tspan[0]

    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun,z,Phi, alph,tk,tk+h)
            tk += h

    # ASSUME all examples are equally weighted
    costL  = torch.mean(z[:,-3])
    costF  = torch.mean(z[:,-2])
    costG  = torch.mean(G(z))
    costHJ = torch.mean(z[:,-1])
    costHJf = 0.0 # not implemented, not needed for OT-Flow

    cs = [costL, costF, costG, costHJ, costHJf]
    # return dot(cs, alph)  , cs
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 5 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z

def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 5 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K

    return z



def integrate(x, net, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0], intermediates=False ):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 5, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+4-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """

    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 4, 0, 0), value=tspan[0])

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
        zFull[:,:,0] = z

        if stepper == 'rk4':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                zFull[:,:,k+1] = stepRK1(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
                tk += h

        return zFull

    else:
        if stepper == 'rk4':
            for k in range(nt):
                z = stepRK4(odefun,z,net, alph,tk,tk+h)
                tk += h
        elif stepper == 'rk1':
            for k in range(nt):
                z = stepRK1(odefun,z,net, alph,tk,tk+h)
                tk += h

        return z

    # return in case of error
    return -1



def G(z):
    """normalizing flows KL-based loss"""
    d = z.shape[1]-4
    l = z[:,d] # log-det

    return -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d],2) / 2  , 1 , keepdims=True ) + l.unsqueeze(1) )


def odefun(x, t, net, alph=[1.0,1.0,1.0,1.0,1.0]):
    """
    the diffeq function for the 4 ODEs in one

    d_t  [x ; l ; v ; hj] = odefun( [x ; l ; v ; hj] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    hj - accumulates least-squares error of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 4

    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    gradPhi, trH = net.trHess(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    df = torch.zeros(nex,1, device=x.device, dtype=x.dtype) # not implemented
    hj = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) # NEEDED if F !=0 :  alph[1]*getDeltaF term
    
    return torch.cat( (dx,dl,dv,df,hj) , 1  )


