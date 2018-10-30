#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:15:49 2018

@author: chen
"""
from collections import OrderedDict
from torchvision import models
from torch import nn
import torch

def get_classifier(
        input_dim = 1920, 
        hidden_dim = 500,
        output_dim = 102,
            ):
    """
    Construct the neural classifier
    """
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_dim, hidden_dim)),
                          ('bn1', nn.BatchNorm1d(hidden_dim)),
                          ('dropout1', nn.Dropout(0.2)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_dim, output_dim)),
                          ('dropout2', nn.Dropout(0.1)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

def get_feature_extractor(name='densenet201'):
    """
    Return pre-trained neural networks for transfer learning.
    """
    if not name in models.__dict__:
        return None
    model = models.__dict__[name](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_model(
        feature_extractor_name,
        hidden_dim = 500,
        output_dim = 102,
        ):
    """
    Load pretrained netwarks and repalcy the classifier for transfer learning.
    """
    model = get_feature_extractor(feature_extractor_name)
    if not model:
        print("Please specify a valid pretraining model name, such as 'densenet201' and 'resnet152'")
        raise Exception("Transfer learning can't find pretrained model.")
    input_dim = model.classifier.in_features
    classifier = get_classifier(input_dim, hidden_dim, output_dim)
    model.classifier = classifier
    return model 


def load_trained_model(path, device_name='cpu'):
    """
    Load pretraind models
    """
    device = torch.device(device_name)
    if device_name == "cpu":
        loaded = torch.load(path, map_location=device_name)
    else:
        loaded = torch.load(path)
    if loaded['pre_model_name'] == 'densenet201':
        model = models.densenet201(pretrained=False)
    else:
        print('This mode is not supported, please specify "densenet201"')
        
    for param in model.parameters():
        param.requires_grad = False   
    
    model.classifier = get_classifier(model.classifier.in_features,
                                     loaded['hidden_dim'],
                                     loaded['output_dim'])
    class2idx = loaded['class2idx']
    cat_to_name = loaded['cat_to_name']
    model.load_state_dict(loaded['model_state_dict'])
    return model.to(device), class2idx, cat_to_name
