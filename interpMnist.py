# interpMnist.py
#
# grab two images, encode them and flow them to rho_1, interpolate between and flow back and decode
# plot many of these interpolations in the latent space
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import argparse
import os
from src.OTFlowProblem import *
import config
import datasets
from datasets.mnist  import getLoader
from src.Autoencoder import *

cf = config.getconfig()
def_resume = 'experiments/cnf/large/pretrained/pretrained_interp_mnist_checkpt.pth'

parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', type=str, default='mnist'
)

parser.add_argument("--nt"        , type=int, default=16, help="number of time steps")
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--resume'    , type=str, default=def_resume)
parser.add_argument('--save'      , type=str, default='image/')
parser.add_argument('--gpu'       , type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    if args.resume is None:
        print("have to provide path to saved model via --resume commandline argument")
        exit(1)

    _ , _ , test_loader = getLoader(args.data, args.batch_size, args.batch_size, augment=False, hasGPU=cf.gpu)

    nt = args.nt
    # --------------------------LOADING------------------------------------
    # reload model
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    m    = checkpt['args'].m
    alph = checkpt['args'].alph
    d    = checkpt['state_dict']['A'].size(1) - 1
    eps  = checkpt['args'].eps
    net = Phi(nTh=2, m=m, d=d, alph=alph)  # the phi aka the value function
    net.load_state_dict(checkpt["state_dict"])

    # get expected type
    prec = net.A.dtype
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)
    net = net.to(prec).to(device)

    # load the trained autoencoder
    autoEnc = Autoencoder(d)
    autoEnc.mu = checkpt['autoencoder']["mu"]
    autoEnc.std = checkpt['autoencoder']["std"]
    autoEnc.load_state_dict(checkpt['autoencoder'], strict=False)  # doesnt load the buffers
    autoEnc = autoEnc.to(prec).to(device)
    # ---------------------------------------------------------------------

    nInterp = 5

    net.eval()
    with torch.no_grad():

        torch.manual_seed(0) # for reproducibility

        images, labels = next(iter(test_loader))

        # vectorize each image
        images = cvt(images.view(images.size(0), -1))

        # grab a few of the class 9
        idx9 = labels == 9
        x9 = images[idx9,:]
        cosIdx = [3,5,6,7,8]
        x0 = x9[cosIdx]
        nSamples = 4
        x0orig = x0[0:nSamples,:]

        # grab one image of an mnist 1 and use it
        idx1 = labels == 1
        x1 = images[idx1,:]
        x0orig[nSamples-1,:] = x1[4,:]
        x0 = autoEnc.encode(x0orig)  # encode
        x0 = (x0 - autoEnc.mu) / (autoEnc.std + eps)  # normalize
        z1 = integrate(x0[:, 0:d], net, [0.0, 1.0], nt, stepper="rk4", alph=net.alph)[0:d] # flow to rho_1
        z1 = z1[:,0:d]

        recastZ = cvt(torch.zeros((nInterp+1)**2, z1.shape[1]))

        # will make a nInterp+1-by-nInterp+1 image with the four corners as the original images
        # upper left, upper right, lower left, lower right
        ul = z1[0, :]
        ur = z1[1, :]
        ll = z1[2, :]
        lr = z1[3, :]

        # assume nInterp = 5
        # hard coded
        # first row
        recastZ[0, :] = ul
        recastZ[1, :] = ul + 0.2 * (ur - ul)
        recastZ[2, :] = ul + 0.4 * (ur - ul)
        recastZ[3, :] = ul + 0.6 * (ur - ul)
        recastZ[4, :] = ul + 0.8 * (ur - ul)
        recastZ[nInterp, :] = ur
        # last row
        recastZ[nInterp*(nInterp+1)   , :] = ll
        recastZ[nInterp*(nInterp+1)+1 , :] = ll + 0.2 * (lr - ll)
        recastZ[nInterp*(nInterp+1)+2 , :] = ll + 0.4 * (lr - ll)
        recastZ[nInterp*(nInterp+1)+3 , :] = ll + 0.6 * (lr - ll)
        recastZ[nInterp*(nInterp+1)+4 , :] = ll + 0.8 * (lr - ll)
        recastZ[(nInterp+1)**2 - 1    , :] = lr

        # for each column, interpolate between the top image and the bottom
        for col in range(nInterp+1):
            top = recastZ[ col , :]
            bot = recastZ[nInterp*(nInterp+1) + col , :]
            for row in range(1,nInterp):
                recastZ[row*(nInterp+1)+col , :] = top + 1.0/nInterp * row * (bot-top)

        gen = integrate(recastZ[:, 0:d], net, [1.0, 0.0], nt, stepper="rk4", alph=net.alph)[:,0:d]
        gen = autoEnc.decode(gen * (autoEnc.std + eps) + autoEnc.mu)

        # place originals in the corner spots
        gen[0, :]  = x0orig[0, :]
        gen[nInterp, :]  = x0orig[1, :]
        gen[nInterp*(1+nInterp), :] = x0orig[2, :]
        gen[(nInterp+1)**2 - 1, :] = x0orig[3, :]

        # plot them
        nex = 48
        fig, axs = plt.subplots(nInterp+1, nInterp+1)
        fig.set_size_inches(6, 6.1)
        fig.suptitle("red boxed values are original; others are interpolated in rho_1 space")
        gen = gen.detach().cpu().numpy()

        k = 0
        for i in range(nInterp+1):
            for j in range(nInterp+1):
                axs[i, j].imshow(gen[k,:].reshape(28,28), cmap='gray')
                # box the originals
                if (i==0 and j==0) or (i==nInterp and j==0) or (i==0 and j==nInterp) or (i==nInterp and j==nInterp):
                    # Create a Rectangle patch
                    rect = patches.Rectangle((0, 0), 27, 27, linewidth=2, edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    axs[i, j].add_patch(rect)
                k+=1

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                axs[i, j].get_yaxis().set_visible(False)
                axs[i, j].get_xaxis().set_visible(False)
                axs[i ,j].set_aspect('equal')

        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        # save figure
        sPath = args.save + 'interpMNISTGen.pdf'
        if not os.path.exists(os.path.dirname(sPath)):
            os.makedirs(os.path.dirname(sPath))
        plt.savefig(sPath, dpi=300)
        plt.close()
        print('figure saved to ', sPath)
