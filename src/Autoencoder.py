# Autoencoder.py
# encoder-decoder used for MNIST experiments
#
# from: https://medium.com/analytics-vidhya/dimension-manipulation-using-autoencoder-in-pytorch-on-mnist-dataset-7454578b018

import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils as utils
import os
from src.plotter import *


# define the encoder-decoder architecture
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        ## encoder ##

        self.d = encoding_dim

        # linear layer (784 -> encoding_dim)
        self.layer1 = nn.Linear(28 * 28, encoding_dim)

        ## decoder ##
        # linear layer (encoding_dim -> input size)
        self.layer2 = nn.Linear(encoding_dim, 28 * 28)

        # register these as buffers
        self.register_buffer('mu', None)
        self.register_buffer('std', None)

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)

        return x

    def encode(self,x):
        # add layer, with relu activation function
        return F.relu(self.layer1(x))

    def decode(self,x):
        # output layer (sigmoid for scaling from 0 to 1)
        return torch.sigmoid(self.layer2(x))



def trainAE(net, train_loader, val_loader, saveDir, sStartTime, argType=torch.float32, device=torch.device('cpu')):
    """

    :param net:          AutoEncoder
    :param train_loader: MNIST loader of training data
    :param val_loader:   MNIST loader of validation data
    :param saveDir:      string, path
    :param sStartTime:   string, start time
    :param argType:      torch type
    :param device:       torch device
    :return:
    """
    print("training auto_encoder")
    
    cvt = lambda x: x.type(argType).to(device, non_blocking=True)
    utils.makedirs(saveDir)

    # specify loss function
    criterion = nn.MSELoss()

    # specify loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    best_loss = float('inf')
    bestParams = None

    # number of epochs to train the model
    n_epochs = 600

    for epoch in range(1, n_epochs + 1):

        # train the encoder-decoder
        net.train()
        train_loss = 0.0
        for data in train_loader:
            # _ stands in for labels, here
            images, _ = data
            # flatten images
            images = images.view(images.size(0), -1)
            images = cvt(images)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # validate the encoder-decoder
        net.eval()
        val_loss = 0.0
        for data in val_loader:
            images, _ = data
            images = images.view(images.size(0), -1)
            images = cvt(images)

            outputs = net(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            val_loss += loss.item() * images.size(0)

        # print avg training statistics...different batch_sizes will scale these differnetly
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        print('Epoch: {} \tTraining Loss: {:.6f} \t Validation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            val_loss
        ))

        # save best set of parameters
        if val_loss < best_loss:
            best_loss = val_loss
            bestParams = net.state_dict()

        # plot
        if epoch % 20 == 0:
            net.eval()
            sSavePath = os.path.join(saveDir, 'figs', sStartTime + '_autoencoder{:d}.png'.format(epoch))
            xRecreate = net(images)
            plotAutoEnc(images, xRecreate, sSavePath)

        # shrink step size
        if epoch % 150 == 0:
            for p in optimizer.param_groups:
                p['lr'] /= 10.0
            print("lr: ", p['lr'])

    d = net.d

    # compute mean and std for normalization
    mu     = torch.zeros((1, d), dtype=argType, device=device)
    musqrd = torch.zeros((1, d), dtype=argType, device=device)
    totImages = 0

    net.load_state_dict(bestParams)

    i = 0
    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # _ stands in for labels, here
            images, _ = data
            images  = images.view(images.size(0), -1)
            images  = cvt(images)
            outputs = net.encode(images)
            nImages = outputs.shape[0]
            totImages += nImages
            mu     += torch.mean(outputs, dim=0, keepdims=True)  # *nImages
            musqrd += torch.mean(outputs ** 2, dim=0, keepdims=True)  # *nImages

            # check quality
            if i == 0:
                sSavePath = os.path.join(saveDir, 'figs', sStartTime + '_autoencoder.png')
                outputs   = (net.encode(images) - 2.34) / 0.005
                xRecreate = net.decode(outputs * 0.005 + 2.34)
                plotAutoEnc(images, xRecreate, sSavePath)

                sSavePath = os.path.join(saveDir, 'figs', sStartTime + '_noise_autoencoder.png')
                xRecreate = net.decode(outputs + 1.0 * torch.randn_like(outputs))
                plotAutoEnc(images, xRecreate, sSavePath)

            i += 1

        mu     = mu / i
        musqrd = musqrd / i
        std    = torch.sqrt(torch.abs(mu ** 2 - musqrd))

        mu.requires_grad  = False
        std.requires_grad = False
        net.mu = mu
        net.std = std

        torch.save({
            'state_dict': net.state_dict(),
        }, os.path.join(saveDir, sStartTime + '_autoenc_checkpt.pth'))

        return net









