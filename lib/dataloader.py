import torch
from torchvision import datasets, transforms
from lib.transform import AddUniformNoise, ToTensor, HorizontalFlip, Transpose, Resize


dataFolder = './data/'

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x

def dataloader(dataset, batch_size, cuda, conditional=-1, im_size=64):

    if dataset == 'mnist':
        data = datasets.MNIST(dataFolder+'MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       AddUniformNoise(),
                       ToTensor()
                   ]))

        test_data = datasets.MNIST(dataFolder+'MNIST', train=False, download=True,
                    transform=transforms.Compose([
                        AddUniformNoise(),
                        ToTensor()
                    ]))

        if conditional >= 0 and conditional <= 9:
            idx = data.targets == conditional
            data.data = data.data[idx, :]
            data.targets = data.targets[idx]
            nTot = torch.sum(idx).item()
            nTrain = int((5.0 / 6.0) * nTot)
            nVal = nTot - nTrain
            train_data, valid_data = torch.utils.data.random_split(data, [nTrain, nVal])

            idx = test_data.targets == conditional
            test_data.data = test_data.data[idx,:]
            test_data.targets = test_data.targets[idx]
        else:
            train_data, valid_data = torch.utils.data.random_split(data, [50000, 10000])

    else:  
        print ('what network ?', dataset)
        sys.exit(1)

    #load data 
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda > -1 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True, **kwargs)
 
    test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, valid_loader, test_loader






if __name__ == '__main__':

    argPrec = torch.float32
    device = torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    train_loader, val_loader, test_loader = dataloader('mnist', 10000, cuda=-1)

    d = 784
    # compute mean and std for MNIST dataloader
    mu     = torch.zeros((1, d), dtype=argPrec, device=device)
    musqrd = torch.zeros((1, d), dtype=argPrec, device=device)
    totImages = 0

    i = 0

    for data in train_loader:
        # _ stands in for labels, here
        images, _ = data
        images  = images.view(images.size(0), -1)
        images  = cvt(images)
        nImages = images.shape[0]
        totImages += nImages
        mu     += torch.mean(images, dim=0, keepdims=True)  # *nImages
        musqrd += torch.mean(images ** 2, dim=0, keepdims=True)  # *nImages
        i += 1

    mu     = mu / i
    musqrd = musqrd / i
    std    = torch.sqrt(torch.abs(mu ** 2 - musqrd))


    print('mu: ', mu)
    print('std: ', std)

