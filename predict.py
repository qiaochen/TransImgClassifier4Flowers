#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:15:49 2018

@author: chen
"""
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from model_utils import load_trained_model
import argparse
import json


def predict(image_path, model, class2idx, topk=5, device_name='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = Image.open(image_path)
    image = process_image(im)
    model.eval()
    idx2class = {v:k for k,v in class2idx.items()}
    with torch.no_grad():
        log_probs = model(torch.from_numpy(image).unsqueeze(0).float().to(device_name))
        probs = torch.exp(log_probs)
        preds, indices = probs.squeeze().topk(topk)
    return preds.cpu().numpy(), [idx2class[idx] for idx in indices.cpu().numpy()], image

def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std  
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
                                     Predict the name of a flower picture.
                                     Basic usage: python predict.py "./Gentian-Tincture.jpg" "./final_model.pth" """
                                     )
    
    parser.add_argument('pic_path')
    parser.add_argument('trained_model')
    parser.add_argument('--top_k', default=1,type=int, help='Return top K most likely classes: python predict.py input checkpoint --top_k 5')
    parser.add_argument('--category_names', default="",type=str, help='Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=False, help="Use GPU for inference: python predict.py input checkpoint --gpu")
    args = parser.parse_args()
    
    final_model_path = args.trained_model
    img_path = args.pic_path
    device_name = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    topk = args.top_k
    
    loaded_model, class2idx, cat_to_name = load_trained_model(final_model_path, device_name=device_name)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    
    probs, classes, image = predict(img_path, loaded_model, device_name = device_name, class2idx=class2idx, topk=topk)
    class_names = [cat_to_name[clz] for clz in classes]
    print("*"*25)
    print("The top {} candidates are:".format(topk))
    for clz, prob in zip(class_names, probs):
        print("Flower: {}, \nProbability: {}.".format(clz, prob))
        print('-'*20)
    
    
