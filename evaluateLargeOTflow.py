# evaluateLargeOTflow.py
# run model on testing set, calculate MMD, and plot
import argparse
import os
import time
import numpy as np
import lib.utils as utils
from lib.utils import count_parameters
from src.plotter import *
from src.OTFlowProblem import *
import h5py
import datasets
from src.mmd import mmd
import config

cf = config.getconfig()
plt.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='miniboone'
)
parser.add_argument('--resume', type=str, default="experiments/cnf/large/pretrained/pretrained_miniboone_checkpt.pth")

parser.add_argument("--nt"  , type=int, default=18, help="number of integration time steps")
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--prec', type=str, default='single', choices=['None', 'single','double'], help="overwrite trained precision")
parser.add_argument('--gpu' , type=int, default=0)
parser.add_argument('--long_version'  , action='store_true')
# default is: args.long_version=False , passing  --long_version will take a long time to run to get values for paper
args = parser.parse_args()

# logger
args.save, sPath = os.path.split(args.resume)
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


def batch_iter(X, batch_size=args.batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]

def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')

def compute_loss(net, x, nt):
    Jc , cs = OTFlowProblem(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs


if __name__ == '__main__':

    if args.long_version:
        sH5ffjord = 'ffjordResults/' + args.data + 'TestFFJORD.h5'
        hf = h5py.File(sH5ffjord, 'r') # open FFJORD results for plotting
        """
        FFJORD results were saved in an h5 file with initial data (copied so that ordering is preserved)
        hf.keys()
        x - the test data from dataset (miniboone, power, etc.)
        fx - f(x) , FFJORD's forward transformation of x to the standard normal
        finvfx - f^{-1} (f(x)) , FFJORD's backward transformation of fx
        invErr - inverse error, avg. norm of difference between x and finvfx ; computed using a weighted avg
        nWeights - number of weights in the FFJORD model
        testTime - how long FFJORD took to compute the testing loss on 1 gpu for the dataset's testing data
        testBatchSize - the batch size used to achieve testTime
        normSamples - 100K samples drawn from the standard normal
        genSamples - f^{-1} (normSamples) , generated points by applying FFJORD backward transformation to the normal dist. pts
        """

        testData      = torch.from_numpy(np.array(hf['x']))
        ffjordFx      = np.array(hf['fx'])
        ffjordFinvfx  = np.array(hf['finvfx'])
        ffjordTime    = np.array(hf['testTime']).item()
        ffjordWeights = np.array(hf['nWeights']).item()
        normSamples   = torch.from_numpy(np.array(hf['normSamples'])) # 10^5 samples
        ffjordGen     = np.array(hf['genSamples'])

    else:
        logger.info("\nABBREVIATED VERSION\n")
        data = load_data(args.data)
        testData = torch.from_numpy(data.tst.x) # x sampled from unknown rho_0
        nSamples = 3000 # 100000
        normSamples = torch.randn(nSamples, testData.shape[1]) # y sampled from rho_1

    logger.info("test data shape: {:}".format(testData.shape))

    nex = testData.shape[0]
    d   = testData.shape[1]
    nt_test = args.nt

    # reload model
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    print(checkpt['args'])
    m       = checkpt['args'].m
    alph    = checkpt['args'].alph
    nTh     = checkpt['args'].nTh 
    net     = Phi(nTh=nTh, m=m, d=d, alph=alph)
    argPrec = checkpt['state_dict']['A'].dtype
    net = net.to(argPrec)
    net.load_state_dict(checkpt["state_dict"])
    net = net.to(device)

    # if specified precision supplied, override the loaded precision
    if args.prec != 'None':
        if args.prec == 'single':
            argPrec = torch.float32 
        if args.prec == 'double':
            argPrec = torch.float64 
        net = net.to(argPrec)

    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    logger.info(net)
    logger.info("----------TESTING---------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,net.alph))
    logger.info("nt_test={:}".format(nt_test))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("Number of testing examples: {}".format(nex))
    logger.info("-------------------------")
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    end = time.time()

    log_msg = (
        '{:4s}        {:9s}  {:9s}  {:11s}  {:9s}'.format(
            'itr', 'loss', 'L (L_2)', 'C (loss)', 'R (HJB)'
        )
    )
    logger.info(log_msg)

    if not cf.gpu:
        # assume debugging and run a subset
        nSamples = 1000
        testData = testData[:nSamples, :]
        normSamples = normSamples[:nSamples, :]
        if args.long_version:
            ffjordFx = ffjordFx[:nSamples, :]
            ffjordFinvfx = ffjordFinvfx[:nSamples, :]
            ffjordGen    = ffjordGen[:nSamples, :]

    net.eval()
    with torch.no_grad():

        # meters to hold testing results
        testLossMeter  = utils.AverageMeter()
        testAlphMeterL = utils.AverageMeter()
        testAlphMeterC = utils.AverageMeter()
        testAlphMeterR = utils.AverageMeter()

        # scale the GAS data set as it was in the training
        if args.data == 'gas':
            print(torch.min(testData),torch.max(testData))
            testData = testData / 5.0

        itr = 1
        for x0 in batch_iter(testData, batch_size=args.batch_size):

            x0 = cvt(x0)
            nex = x0.shape[0]
            test_loss, test_cs = compute_loss(net, x0, nt=nt_test)
            testLossMeter.update(test_loss.item(), nex)
            testAlphMeterL.update(test_cs[0].item(), nex)
            testAlphMeterC.update(test_cs[1].item(), nex)
            testAlphMeterR.update(test_cs[2].item(), nex)
            log_message = 'batch {:4d}: {:9.3e}  {:9.3e}  {:11.5e}  {:9.3e}'.format(
                itr, test_loss, test_cs[0], test_cs[1], test_cs[2]
            )
            logger.info(log_message)  # print batch
            itr+=1

        # add to print message
        log_message = '[TEST]      {:9.3e}  {:9.3e}  {:11.5e}  {:9.3e} '.format(
            testLossMeter.avg, testAlphMeterL.avg, testAlphMeterC.avg, testAlphMeterR.avg
        )

        logger.info(log_message) # print total
        logger.info("Testing Time:          {:.2f} seconds with {:} parameters".format( time.time() - end, count_parameters(net) ))
        if args.long_version:
            logger.info("FFJORD's Testing Time: {:.2f} seconds with {:} parameters".format( ffjordTime , ffjordWeights ))


        # computing inverse
        logger.info("computing inverse...")
        nGen = normSamples.shape[0]

        modelFx     = np.zeros(testData.shape)
        modelFinvfx = np.zeros(testData.shape)
        modelGen    = np.zeros(normSamples.shape)

        idx = 0
        for i , x0 in enumerate(batch_iter(testData, batch_size=args.batch_size)):
            x0     = cvt(x0)
            fx     = integrate(x0[:, 0:d], net, [0.0, 1.0], nt_test, stepper="rk4", alph=net.alph)
            finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_test, stepper="rk4", alph=net.alph)

            # consolidate fx and finvfx into one spot
            batchSz = x0.shape[0]
            modelFx[ idx:idx+batchSz , 0:d ]     = fx[:,0:d].detach().cpu().numpy()
            modelFinvfx[ idx:idx+batchSz , 0:d ] = finvfx[:,0:d].detach().cpu().numpy()
            idx = idx + batchSz

        logger.info("model inv error:  {:.3e}".format(np.linalg.norm(testData.numpy() - modelFinvfx) / nex))
        if args.long_version:
            logger.info("FFJORD inv error: {:.3e}".format( np.array(hf['invErr']).item()  ))

        # this portion can take a long time
        # generate samples
        logger.info("generating samples...")
        idx = 0
        for i, y in enumerate(batch_iter(normSamples, batch_size=args.batch_size)):
            y = cvt(y)
            finvy = integrate(y[:, 0:d], net, [1.0, 0.0], nt_test, stepper="rk4",alph=net.alph)

            batchSz = y.shape[0]
            modelGen[ idx:idx+batchSz , 0:d ] = finvy[:,0:d].detach().cpu().numpy()
            idx = idx + batchSz

        # plotting
        sPath = os.path.join(args.save, 'figs', sPath[:-12] + '_test')
        if not os.path.exists(os.path.dirname(sPath)):
            os.makedirs(os.path.dirname(sPath))

        testData = testData.detach().cpu().numpy()  # make to numpy
        normSamples = normSamples.detach().cpu().numpy()

        if not args.long_version: # when running abbreviated style, use smaller sample sizes to compute mmd so its quicker
            nSamples = min(testData.shape[0], modelGen.shape[0], 3000)  # number of samples for the MMD
            testSamps = testData[0:nSamples, :]
            modelSamps = modelGen[0:nSamples, 0:d]
        else:
            testSamps = testData[0:nSamples, :]
            modelSamps = modelGen[:, 0:d]

        if args.data=='gas':
            # scale back
            modelSamps = modelSamps * 5.0
            testSamps  = testSamps  * 5.0
            testData   = testData * 5.0
            modelGen   = modelGen * 5.0

        print("MMD( ourGen   , rho_0 ),  num(ourGen)={:d}    , num(rho_0)={:d} : {:.5e}".format( modelSamps.shape[0]  , testSamps.shape[0] , mmd(modelSamps  , testSamps )))
        if args.long_version:
            ffjordSamps = ffjordGen
            print("MMD( FFJORDGen, rho_0 ),  num(FFJORDGen)={:d} , num(rho_0)={:d} : {:.5e}".format( ffjordSamps.shape[0] , testSamps.shape[0] , mmd(ffjordSamps , testSamps )))

        logger.info("plotting...")
        nBins = 33
        LOW = -4
        HIGH = 4

        if args.data == 'gas':
            # the gas data set has different bounds
            LOWrho0 = -2
            HIGHrho0 = 2
            nBins = 33
        else:
            LOWrho0 = LOW
            HIGHrho0 = HIGH

        bounds = [[LOW, HIGH], [LOW, HIGH]]
        boundsRho0 = [[LOWrho0, HIGHrho0], [LOWrho0, HIGHrho0]]

        for d1 in range(0, d-1, 2):  # plot 2-D slices of the multivariate distribution
            d2 = d1 + 1
            fig, axs = plt.subplots(2,3)    # (2, 2)
            fig.set_size_inches(20,12)  # (14,10)
            fig.suptitle(args.data + "  dims: {:d} vs {:d}".format(d1, d2))

            # hist, xbins, ybins, im = axs[0, 0].hist2d(x.numpy()[:,0],x.numpy()[:,1], range=[[LOW, HIGH], [LOW, HIGH]], bins = nBins)
            im1, _, _, map1 = axs[0, 0].hist2d(testData[:, d1], testData[:, d2], range=boundsRho0, bins=nBins)
            axs[0, 0].set_title(r'$x \sim \rho_0(x)$')

            im2, _, _, map2 = axs[0, 1].hist2d(modelFx[:, d1], modelFx[:, d2], range=bounds, bins=nBins)
            axs[0, 1].set_title(r'$f(x)$')

            im3, _, _, map3 = axs[1, 0].hist2d(normSamples[:, d1], normSamples[:, d2], range=bounds, bins=nBins)
            axs[1, 0].set_title(r'$y \sim \rho_1(y)$')

            im4, _, _, map4 = axs[1, 1].hist2d(modelGen[:, d1],modelGen[:, d2], range=boundsRho0, bins=nBins)
            axs[1, 1].set_title(r'$f^{-1}(y)$')

            if args.long_version:
                im5, _, _, map5 = axs[0, 2].hist2d(ffjordFx[:, d1], ffjordFx[:, d2], range=bounds, bins=nBins)
                axs[0, 2].set_title(r'FFJORD $f(x)$')

                im6, _, _, map6 = axs[1, 2].hist2d(ffjordGen[:, d1], ffjordGen[:, d2], range=boundsRho0, bins=nBins)
                axs[1, 2].set_title(r'FFJORD $f^{-1}(y)$')
            else:
                placeholder = 100.0*np.ones_like(testData[:, 0])
                im5, _, _, map5 = axs[0, 2].hist2d(placeholder, placeholder, range=bounds, bins=nBins)
                axs[0, 2].set_title('placeholder')

                im6, _, _, map6 = axs[1, 2].hist2d(placeholder, placeholder, range=boundsRho0, bins=nBins)
                axs[1, 2].set_title('placeholder')


            # each has its own colorbar
            fig.colorbar(map1, cax=fig.add_axes([0.35 , 0.53, 0.01, 0.35]) )
            fig.colorbar(map2, cax=fig.add_axes([0.625, 0.53, 0.01, 0.35]) )
            fig.colorbar(map3, cax=fig.add_axes([0.35 , 0.11, 0.01, 0.35]) )
            fig.colorbar(map4, cax=fig.add_axes([0.625, 0.11, 0.01, 0.35]) )
            fig.colorbar(map5, cax=fig.add_axes([0.90 , 0.53, 0.01, 0.35]) )
            fig.colorbar(map6, cax=fig.add_axes([0.90 , 0.11, 0.01, 0.35]) )


            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    axs[i, j].get_yaxis().set_visible(False)
                    axs[i, j].get_xaxis().set_visible(False)
                    axs[i, j].set_aspect('equal')

            # plt.show()
            plt.savefig(sPath + "_{:d}v{:d}.pdf".format(d1, d2), dpi=400)
            plt.close()

    if args.long_version:
        hf.close() # close the h5 file
    logger.info('Testing has finished.  ' + sPath)







