# trainLargeOTflow.py
# train OT-Flow for the large density estimation data sets
import argparse
import os
import time
import datetime
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters

from src.plotter import plot4
from src.OTFlowProblem import *
from src.Phi import *
import config
import datasets

cf = config.getconfig()

if cf.gpu:
    def_viz_freq = 200
    def_batch    = 2000
    def_niter    = 8000
    def_m        = 256
    def_val_freq = 0
else: # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 20
    def_val_freq = 20
    def_batch    = 200
    def_niter    = 2000
    def_m        = 16

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300','mnist'], type=str, default='miniboone'
)

parser.add_argument("--nt"    , type=int, default=6, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=10, help="number of time steps for validation")
parser.add_argument('--alph'  , type=str, default='1.0,100.0,15.0')
parser.add_argument('--m'     , type=int, default=def_m)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--lr'       , type=float, default=0.01)
parser.add_argument("--drop_freq", type=int  , default=0, help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument("--lr_drop"  , type=float, default=10.0, help="how much to decrease learning rate (divide by)")
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--prec'      , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--niters'    , type=int, default=def_niter)
parser.add_argument('--batch_size', type=int, default=def_batch)
parser.add_argument('--test_batch_size', type=int, default=def_batch)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--early_stopping', type=int, default=20)

parser.add_argument('--save', type=str, default='experiments/cnf/large')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=def_val_freq) # validation frequency needs to be less than viz_freq or equal to viz_freq
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# add timestamp to save path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32



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


# decrease the learning rate based on validation
ndecs = 0
n_vals_wo_improve=0
def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs = 2
    else:
        ndecs += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs


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

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    data = load_data(args.data)
    data.trn.x = torch.from_numpy(data.trn.x)
    print(data.trn.x.shape)
    data.val.x = torch.from_numpy(data.val.x)

    if args.data == 'gas': # scale the gas data smaller for stability
        data.trn.x = data.trn.x / 5.0
        data.val.x = data.val.x / 5.0

    # hyperparameters of model
    d   = data.trn.x.shape[1]
    nt  = args.nt
    nt_val = args.nt_val
    nTh = args.nTh
    m   = args.m

    # set up neural network to model potential function Phi
    net = Phi(nTh=nTh, m=m, d=d, alph=args.alph)
    net = net.to(prec).to(device)


    # resume training on a model that's already had some training
    if args.resume is not None:
        # reload model
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        m       = checkpt['args'].m
        alph    = args.alph # overwrite saved alpha
        nTh     =  checkpt['args'].nTh
        args.hutch = checkpt['args'].hutch
        net     = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the phi aka the value function
        prec = checkpt['state_dict']['A'].dtype
        net     = net.to(prec)
        net.load_state_dict(checkpt["state_dict"])
        net     = net.to(device)

    if args.val_freq == 0:
        # if val_freq set to 0, then validate after every epoch
        args.val_freq = math.ceil(data.trn.x.shape[0]/args.batch_size)

    # ADAM optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(net)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,net.alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    begin = time.time()
    end = begin
    best_loss = float('inf')
    best_cs = [0.0]*3
    bestParams = None

    log_msg = (
        '{:5s}  {:6s}  {:7s}   {:9s}  {:9s}  {:9s}  {:9s}     {:9s}  {:9s}  {:9s}  {:9s} '.format(
            'iter', ' time','lr','loss', 'L (L2)', 'C (loss)', 'R (HJB)', 'valLoss', 'valL', 'valC', 'valR',
        )
    )
    logger.info(log_msg)

    timeMeter = utils.AverageMeter()

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    net.train()
    itr = 1
    while itr < args.niters:
        # train
        for x0 in batch_iter(data.trn.x, shuffle=True):   
            x0 = cvt(x0)
            optim.zero_grad()

            # clip parameters
            for p in net.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)

            currParams = net.state_dict()
            loss,cs  = compute_loss(net, x0, nt=nt)
            loss.backward()
          
            optim.step()
            timeMeter.update(time.time() - end)
            
            log_message = (
                '{:05d}  {:6.3f}  {:7.1e}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                    itr, timeMeter.val, optim.param_groups[0]['lr'], loss, cs[0], cs[1], cs[2]
                )
            )

            if torch.isnan(loss): # catch NaNs when hyperparameters are poorly chosen
                logger.info(log_message)
                logger.info("NaN encountered....exiting prematurely")
                logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                logger.info('File: ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(
                        args.data, int(net.alph[1]), int(net.alph[2]), m)
                )
                exit(1)

            # validation
            if itr % args.val_freq == 0 or itr == args.niters:
                net.eval()
                with torch.no_grad():

                    valLossMeter = utils.AverageMeter()
                    valAlphMeterL = utils.AverageMeter()
                    valAlphMeterC = utils.AverageMeter()
                    valAlphMeterR = utils.AverageMeter()

                    for x0 in batch_iter(data.val.x, batch_size=test_batch_size):
                        x0 = cvt(x0)
                        nex = x0.shape[0]
                        val_loss, val_cs = compute_loss(net, x0, nt=nt_val)
                        valLossMeter.update(val_loss.item(), nex)
                        valAlphMeterL.update(val_cs[0].item(), nex)
                        valAlphMeterC.update(val_cs[1].item(), nex)
                        valAlphMeterR.update(val_cs[2].item(), nex)


                    # add to print message
                    log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                        valLossMeter.avg, valAlphMeterL.avg, valAlphMeterC.avg, valAlphMeterR.avg
                    )

                    # save best set of parameters
                    if valLossMeter.avg < best_loss:
                        n_vals_wo_improve = 0
                        best_loss = valLossMeter.avg
                        best_cs = [  valAlphMeterL.avg, valAlphMeterC.avg, valAlphMeterR.avg ]
                        utils.makedirs(args.save)
                        bestParams = net.state_dict()
                        torch.save({
                            'args': args,
                            'state_dict': bestParams,
                        }, os.path.join(args.save, start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(net.alph[1]),int(net.alph[2]),m)))
                    else:
                        n_vals_wo_improve+=1

                    net.train()
                    log_message += ' no improve: {:d}/{:d}'.format(n_vals_wo_improve, args.early_stopping)
            logger.info(log_message) # print iteration

            # create plots for assessment mid-training
            if itr % args.viz_freq == 0:
                with torch.no_grad():
                    net.eval()
                    currState = net.state_dict()
                    net.load_state_dict(bestParams)

                    # plot one batch 
                    p_samples = cvt(data.val.x[0:test_batch_size,:])
                    nSamples = p_samples.shape[0]
                    y = cvt(torch.randn(nSamples,d)) # sampling from rho_1 / standard normal

                    sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                    plot4(net, p_samples, y, nt_val, sPath, sTitle='loss {:.2f}  ,  C {:.2f}'.format(best_loss, best_cs[1] ))

                    net.load_state_dict(currState)
                    net.train()

            if args.drop_freq == 0: # if set to the code setting 0 , the lr drops based on validation
                if n_vals_wo_improve > args.early_stopping:
                    if ndecs>2:
                        logger.info("early stopping engaged")
                        logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                        logger.info('File: ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(
                            args.data, int(net.alph[1]), int(net.alph[2]), m)
                          )
                        exit(0)
                    else:
                        update_lr(optim, n_vals_wo_improve)
                        n_vals_wo_improve = 0
            else:
                # shrink step size
                if itr % args.drop_freq == 0:
                    for p in optim.param_groups:
                        p['lr'] /= args.lr_drop
                    print("lr: ", p['lr'])

            itr += 1
            end = time.time()
            # end batch_iter

    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(net.alph[1]),int(net.alph[2]),m))






