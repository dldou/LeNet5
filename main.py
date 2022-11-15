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
from torchsummary import summary



if __name__ == "__main__":

    #Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #Model
    nof_classes = 10
    LeNet5 = LeNet_5(nof_classes)
    print(summary(LeNet5, (1, 28, 28)))

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

    #Hyper parameters
    nof_epochs = 1
    lr = 0.0005
    optimizer = torch.optim.Adam(LeNet5.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    #File's path to save the parameters of the model
    file_path='/content/model_parameters.csv'

    #Train
    LeNet5 = train_model(model=LeNet5, train_loader=train_loader, test_loader=test_loader, 
                         nof_epochs=nof_epochs, optimizer=optimizer, learning_rate=lr, criterion=criterion, 
                         file_path_save_model=file_path)
    
    plot_inference(LeNet5, mnist_testset, device)