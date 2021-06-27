"""
This module produces a data loader for MNIST
"""
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

def create_loader(size,batch_size,digits,transform=None,train=True):
    """
    Produce a dataloader of MNIST respecting the original dataset's 
    distribution acrros classes. 
    
    Args: 
        size (scalar int): 
            the size of the desired dataset.
        batch_size (scalar int): 
            size of mini-batches.
        digits (list of ints between 0 and 9): 
            digits to consider 
        transform (torchvision.transforms, optional): 
            Tranforms to apply to data samples. Defaults to None: images 
            are cast to tensor and normalized given MNIST statistics.
        train (bool, optional): 
            whether to use the training or test portions of the dataset. 
            Defaults to True.
    Returns: 
        (torch.utils.data.DataLoader): Resampled MNIST dataset 
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
    mnist = datasets.MNIST(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
        )
    
    # Keep only samples labeled as one of "digits"
    target = mnist.targets
    all_digits = torch.unique(target, sorted=True)
    class_sample_count = torch.tensor(
        [(target == t).sum() if t in digits else np.inf for t in all_digits])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in target])
    
    # Balanced sampling
    sampler = WeightedRandomSampler(samples_weight, size, replacement=False)
    loader = DataLoader(
        mnist,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=12,
        pin_memory=True
        )
    
    return loader