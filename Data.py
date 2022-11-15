#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:01:24 2022

@author: dldou
"""

#Import MNIST (handwritten digits dataset)
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy

###Import the dataset###
transform = transforms.Compose([transforms.CenterCrop(28),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)
                                ])

#Train dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#Dataloader used to shuffle and create batch
train_loader   = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)

#Test dataset
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader   = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=True)

