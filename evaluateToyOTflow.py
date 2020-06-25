# evaluateToyOTflow.py
# plotting toy CNF results
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('agg') # for linux server with no tkinter
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams.update({'font.size': 22})

import argparse
import os
import time
import datetime
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from src.OTFlowProblem import *
from src.mmd import *


def_resume = 'experiments/cnf/toy/pretrained/pretrained_swissroll_alph30_15_m32_checkpt.pth'

parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='swissroll'
)
parser.add_argument("--nt"      , type=int, default=12, help="number of time steps")
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--resume'  , type=str, default=def_resume)
parser.add_argument('--save'    , type=str, default='image/')
parser.add_argument('--gpu'     , type=int, default=0)
args = parser.parse_args()

# logger
_ , sPath = os.path.split(args.resume)
utils.makedirs(args.save)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# loss function
def compute_loss(net, x, nt):
    Jc , cs = OTFlowProblem(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs

if __name__ == '__main__':

    # reload model
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    m       = checkpt['args'].m
    alph    = checkpt['args'].alph
    nTh     = checkpt['args'].nTh
    d       = checkpt['state_dict']['A'].size(1) - 1
    net     = Phi(nTh=nTh, m=m, d=d, alph=alph)
    prec    = checkpt['state_dict']['A'].dtype
    net     = net.to(prec)
    net.load_state_dict(checkpt['state_dict'])
    net     = net.to(device)

    args.data = checkpt['args'].data

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    nSamples = args.batch_size
    p_samples = cvt(torch.Tensor(toy_data.inf_train_gen(args.data, batch_size=nSamples)))
    y         = cvt(torch.randn(nSamples,d))

    net.eval()
    with torch.no_grad():

        test_loss, test_cs = compute_loss(net, p_samples, args.nt)

        # sample_fn, density_fn = get_transforms(model)
        modelFx     = integrate(p_samples[:, 0:d], net, [0.0, 1.0], args.nt, stepper="rk4", alph=net.alph)
        modelFinvfx = integrate(modelFx[:, 0:d]  , net, [1.0, 0.0], args.nt, stepper="rk4", alph=net.alph)
        modelGen    = integrate(y[:, 0:d]        , net, [1.0, 0.0], args.nt, stepper="rk4", alph=net.alph)

        print("          {:9s}  {:9s}  {:11s}  {:9s}".format( "loss", "L (L_2)", "C (loss)", "R (HJB)"))
        print("[TEST]:   {:9.3e}  {:9.3e}  {:11.5e}  {:9.3e}".format(test_loss, test_cs[0], test_cs[1], test_cs[2]))

        print("Using ", utils.count_parameters(net), " parameters")
        invErr = (torch.norm(p_samples-modelFinvfx[:,:d]) / p_samples.size(0)).item()
        print("inv error: ", invErr )

        modelGen = modelGen[:, 0:d].detach().cpu().numpy()
        p_samples = p_samples.detach().cpu().numpy()

        nBins = 80
        LOW = -4
        HIGH = 4
        extent = [[LOW, HIGH], [LOW, HIGH]]

        d1 = 0
        d2 = 1

        # density function of the standard normal
        def normpdf(x):
            mu = torch.zeros(1, d, device=x.device, dtype=x.dtype)
            cov = torch.ones(1, d, device=x.device, dtype = x.dtype)  # diagonal of the covariance matrix

            denom = (2 * math.pi) ** (0.5 * d) * torch.sqrt(torch.prod(cov))
            num = torch.exp(-0.5 * torch.sum((x - mu) ** 2 / cov, 1, keepdims=True))
            return num / denom

        print("plotting...")
        # ----------------------------------------------------------------------------------------------------------
        # Plot Density
        # ----------------------------------------------------------------------------------------------------------
        title = "$density$"

        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(1, 1, 1, aspect="equal")

        npts = 100

        side = np.linspace(LOW, HIGH, npts)
        xx, yy = np.meshgrid(side, side)
        x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
        with torch.no_grad():
            x = cvt(torch.from_numpy(x))
            nt_val = args.nt
            z = integrate(x, net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
            logqx = z[:, d]
            z = z[:, 0:d]

        qz = np.exp(logqx.cpu().numpy()).reshape(npts, npts)
        normpdfz = normpdf(z)
        rho0 = normpdfz.cpu().numpy().reshape(npts, npts) * qz

        im = plt.pcolormesh(xx, yy, rho0)
        vmin = np.min(rho0)
        vmax = np.max(rho0)
        im.set_clim(vmin, vmax)
        ax.axis('off')

        sSaveLoc = os.path.join(args.save, sPath[:-12] + '_density.png')
        plt.savefig(sSaveLoc,bbox_inches='tight')
        plt.close(fig)

        # ----------------------------------------------------------------------------------------------------------
        # Plot Original Samples
        # ----------------------------------------------------------------------------------------------------------

        x0 = toy_data.inf_train_gen(args.data, batch_size=nSamples)  # load data batch
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(1, 1, 1, aspect="equal")
        h2, _, _, map2 = ax.hist2d(x0[:, d1], x0[:, d2], range=extent, bins=nBins)
        # vmax: 15 for swissroll, 8gaussians, moons, 20 for pinwheel, 10 for circles, 8 for checkerboards
        h2 = h2 / (nSamples)
        im2 = ax.imshow(h2);
        ax.axis('off')
        im2.set_clim(vmin, vmax)
        sSaveLoc = os.path.join(args.save, sPath[:-12] + '_rho0Samples.png')
        plt.savefig(sSaveLoc,bbox_inches='tight')
        plt.close(fig)

        # ----------------------------------------------------------------------------------------------------------
        # Plot Generated Samples
        # ----------------------------------------------------------------------------------------------------------
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(1, 1, 1, aspect="equal")
        y = cvt(torch.randn(nSamples, d))
        genModel = integrate(y[:, 0:d], net, [1.0, 0.0], args.nt, stepper="rk4", alph=net.alph)
        h3, _, _, map3 = ax.hist2d(genModel.detach().cpu().numpy()[:, d1], genModel.detach().cpu().numpy()[:, d2],
                  range=extent, bins=nBins)
        h3 = h3/(nSamples)
        im3 = ax.imshow(h3)
        im3.set_clim(vmin, vmax)
        ax.axis('off')
        sSaveLoc = os.path.join(args.save, sPath[:-12] + '_genSamples.png')
        plt.savefig(sSaveLoc,bbox_inches='tight')
        plt.close(fig)
        print("finished plotting to folder", args.save)

    print("testing complete")






