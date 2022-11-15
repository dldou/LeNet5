#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:04:30 2022

@author: dldou
"""


from LeNet5_model import *
from LeNet5_utils import *
from Data import *

import torch

if __name__ == "__main__":

    #Model
    nof_classes = 10
    LeNet5 = LeNet_5(nof_classes)

    #Hyper parameters
    nof_epochs = 20
    lr = 0.1
    optimizer = torch.optim.Adam(LeNet5.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    #File's path to save the parameters of the model
    file_path='/content/model_parameters.csv'

    #Train
    train_model(model=LeNet5, train_loader=train_loader, test_loader=test_loader, 
                nof_epochs=nof_epochs, optimizer=optimizer, learning_rate=lr, criterion=criterion, 
                file_path_save_model=file_path)