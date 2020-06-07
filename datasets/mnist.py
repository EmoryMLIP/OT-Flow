# mnist.py

from torchvision import transforms, datasets
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def getLoader(name, batch, test_batch, augment=False, hasGPU=False, conditional=-1):

    if name == 'mnist':
        val_size = 1.0/6.0
        random_seed = 0

        # define transforms
        normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        kwargs = {'num_workers': 0, 'pin_memory': True} if hasGPU else {}

        # load the dataset
        # from https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
        data = datasets.MNIST(root='../data', train=True,
                                       download=True, transform=train_transform)

#         val_dataset = datasets.MNIST(root='../data', train=True,
#                                        download=True, transform=val_transform)

        test_data = datasets.MNIST(root='../data',
                        train=False,download=True,transform=val_transform)
        
        
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


#         num_train = len(train_dataset)
#         indices = list(range(num_train))
#         split = int(np.floor(val_size * num_train))

        # set up random samplers
#         np.random.seed(random_seed)
#         np.random.shuffle(indices)
#         train_idx, val_idx = indices[split:], indices[:split]
#         train_sampler = SubsetRandomSampler(train_idx)
#         val_sampler   = SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(train_data,
                        batch_size=batch, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(valid_data,
                        batch_size=test_batch, shuffle=False, **kwargs)


        test_loader = torch.utils.data.DataLoader(test_data, shuffle=False,
                        batch_size=test_batch, **kwargs)

        return train_loader, val_loader, test_loader




    else:
        raise ValueError('Unknown dataset')
        exit(1)








