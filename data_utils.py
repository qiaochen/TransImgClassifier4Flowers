#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:44:11 2018

@author: chen
"""
from torchvision import transforms, datasets
import torch

def prepare_data_loaders(train_dir, 
                         valid_dir, 
                         test_dir,
                         batch_size=64
                         ):
    """
    This function loads datasets, preprocessing the input and wrap
    The training, testing and developping sets with DataLoaders
    params:
    - train_dir: str
    - valid_dir: str
    - test_dir: str
    - batch_size: int, the mini-batch size of training
    return:
    - dict for dataloaders, class to index mapping
    """
    trs_means = (0.485, 0.456, 0.406)
    trs_stds = (0.229, 0.224, 0.225)
    
    # define image transformations
    train_transforms = transforms.Compose([
                       transforms.RandomRotation(30),
                       transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(trs_means, trs_stds)])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(trs_means, trs_stds)])
    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    dev_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
#    image_datasets = {'train': train_data, 'valid': dev_data, 'test': test_data}
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    devloader = torch.utils.data.DataLoader(dev_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train': trainloader, 'valid':devloader, 'test':testloader}
    return dataloaders, train_data.class_to_idx