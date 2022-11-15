#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:32:16 2022

@author: dldou
"""

import torch

#Model
class LeNet_5(torch.nn.Module):

    def __init__(self, nof_classes):
        #Heritage from nn.Module
        super(LeNet_5, self).__init__()

        #Features extractor part
        self.feat_extractor_part = torch.nn.Sequential(
            #Conv2D
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=1, stride=1, bias=True),
            torch.nn.ReLU(),
            #Sub-Sampling (average pooling)
            torch.nn.AvgPool2d(kernel_size=(2,2), stride=2, padding=0),
            #Conv2D
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), padding=0, stride=1, bias=True),
            torch.nn.ReLU(),
            #Sub-Sampling (average pooling)
            torch.nn.AvgPool2d(kernel_size=(2,2), stride=2, padding=1),
            #Conv2D
            torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), padding=0, stride=1, bias=True),
            torch.nn.ReLU(),
        )

        self.classification_part = torch.nn.Sequential(
            #Fully connected
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=nof_classes),
            torch.nn.Softmax(),
        )

    def forward(self, x):

        x = self.feat_extractor_part(x)
        x = torch.flatten(x, 1)
        x = self.classification_part(x)
        
        return x

