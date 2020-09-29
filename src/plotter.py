# plotter.py
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg') # for linux server with no tkinter
# matplotlib.use('Agg') # assume no tkinter
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
from src.OTFlowProblem import *
import numpy as np
import os
import h5py
import datasets
from torch.nn.functional import pad
from matplotlib import colors # for evaluateLarge



def plot4(net, x, y, nt_val, sPath, sTitle="", doPaths=False):
    """
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    nSamples = x.shape[0]


    fx = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)
    genModel = integrate(y[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph)

    invErr = torch.norm(x[:,0:d] - finvfx[:,0:d]) / x.shape[0]

    nBins = 33
    LOWX  = -4
    HIGHX = 4
    LOWY  = -4
    HIGHY = 4

    if d > 50: # assuming bsds
        # plot dimensions d1 vs d2 
        d1=0
        d2=1
        LOWX  = -0.15   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 0.15
        LOWY  = -0.15
        HIGHY = 0.15
    if d > 700: # assuming MNIST
        d1=0
        d2=1
        LOWX  = -10   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX = 10
        LOWY  = -10
        HIGHY = 10
    elif d==8: # assuming gas
        LOWX  = -2   # note: there's a hard coded 4 and -4 in axs 2
        HIGHX =  2
        LOWY  = -2
        HIGHY =  2
        d1=2
        d2=3
        nBins = 100
    else:
        d1=0
        d2=1

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(sTitle + ', inv err {:.2e}'.format(invErr))

    # hist, xbins, ybins, im = axs[0, 0].hist2d(x.numpy()[:,0],x.numpy()[:,1], range=[[LOW, HIGH], [LOW, HIGH]], bins = nBins)
    im1 , _, _, map1 = axs[0, 0].hist2d(x.detach().cpu().numpy()[:, d1], x.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins=nBins)
    axs[0, 0].set_title('x from rho_0')
    im2 , _, _, map2 = axs[0, 1].hist2d(fx.detach().cpu().numpy()[:, d1], fx.detach().cpu().numpy()[:, d2], range=[[-4, 4], [-4, 4]], bins = nBins)
    axs[0, 1].set_title('f(x)')
    im3 , _, _, map3 = axs[1, 0].hist2d(finvfx.detach().cpu().numpy()[: ,d1] ,finvfx.detach().cpu().numpy()[: ,d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 0].set_title('finv( f(x) )')
    im4 , _, _, map4 = axs[1, 1].hist2d(genModel.detach().cpu().numpy()[:, d1], genModel.detach().cpu().numpy()[:, d2], range=[[LOWX, HIGHX], [LOWY, HIGHY]], bins = nBins)
    axs[1, 1].set_title('finv( y from rho1 )')

    fig.colorbar(map1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(map2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(map3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(map4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )


    # plot paths
    if doPaths:
        forwPath = integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)
        backPath = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_val, stepper="rk4", alph=net.alph, intermediates=True)

        # plot the forward and inverse trajectories of several points; white is forward, red is inverse
        nPts = 10
        pts = np.unique(np.random.randint(nSamples, size=nPts))
        for pt in pts:
            axs[0, 0].plot(forwPath[pt, 0, :].detach().cpu().numpy(), forwPath[pt, 1, :].detach().cpu().numpy(), color='white', linewidth=4)
            axs[0, 0].plot(backPath[pt, 0, :].detach().cpu().numpy(), backPath[pt, 1, :].detach().cpu().numpy(), color='red', linewidth=2)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            # axs[i, j].get_yaxis().set_visible(False)
            # axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()



def plotAutoEnc(x, xRecreate, sPath):

    # assume square image
    s = int(math.sqrt(x.shape[1]))


    nex = 8

    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].reshape(s, s).detach().cpu().numpy())


    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plotAutoEnc3D(x, xRecreate, sPath):

    nex = 8

    fig, axs = plt.subplots(4, nex//2)
    fig.set_size_inches(9, 9)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nex//2):
        axs[0, i].imshow(x[i,:].permute(1,2,0).detach().cpu().numpy())
        axs[1, i].imshow(x[ nex//2 + i , : ].permute(1,2,0).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].permute(1,2,0).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[ nex//2 + i , : ].permute(1,2,0).detach().cpu().numpy())


    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()



def plotImageGen(x, xRecreate, sPath):

    # assume square image
    s = int(math.sqrt(x.shape[1]))

    nex = 80
    nCols = nex//5


    fig, axs = plt.subplots(7, nCols)
    fig.set_size_inches(16, 7)
    fig.suptitle("first 2 rows originals. Rows 3 and 4 are generations.")

    for i in range(nCols):
        axs[0, i].imshow(x[i,:].reshape(s,s).detach().cpu().numpy())
        # axs[1, i].imshow(x[ nex//3 + i , : ].reshape(s,s).detach().cpu().numpy())
        # axs[2, i].imshow(x[ 2*nex//3 + i , : ].reshape(s,s).detach().cpu().numpy())
        axs[2, i].imshow(xRecreate[i,:].reshape(s,s).detach().cpu().numpy())
        axs[3, i].imshow(xRecreate[nCols + i,:].reshape(s,s).detach().cpu().numpy())
        
        axs[4, i].imshow(xRecreate[2*nCols + i,:].reshape(s,s).detach().cpu().numpy())
        axs[5, i].imshow(xRecreate[3*nCols + i , : ].reshape(s, s).detach().cpu().numpy())
        axs[6, i].imshow(xRecreate[4*nCols + i , : ].reshape(s, s).detach().cpu().numpy())

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plot4mnist(x, sPath, sTitle=""):
    """
    x - tensor (>4, 28,28)
    """
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(sTitle)

    im1 = axs[0, 0].imshow(x[0,:,:].detach().cpu().numpy())
    im2 = axs[0, 1].imshow(x[1,:,:].detach().cpu().numpy())
    im3 = axs[1, 0].imshow(x[2,:,:].detach().cpu().numpy())
    im4 = axs[1, 1].imshow(x[3,:,:].detach().cpu().numpy())

    fig.colorbar(im1, cax=fig.add_axes([0.47, 0.53, 0.02, 0.35]) )
    fig.colorbar(im2, cax=fig.add_axes([0.89, 0.53, 0.02, 0.35]) )
    fig.colorbar(im3, cax=fig.add_axes([0.47, 0.11, 0.02, 0.35]) )
    fig.colorbar(im4, cax=fig.add_axes([0.89, 0.11, 0.02, 0.35]) )

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i ,j].set_aspect('equal')

    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()





