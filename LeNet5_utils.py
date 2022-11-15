#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:57:31 2022

@author: dldou
"""

#Contains utils: training function + saveModel
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random

def saveModel(file_path, model):
    torch.save(model.state_dict(), file_path)

def train_model(model, train_loader, test_loader, 
                nof_epochs, optimizer, learning_rate, criterion, 
                file_path_save_model):

    #Which device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Convert and send model parameters and buffers to GPU 
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(nof_epochs):

        epoch_loss     = 0.0
        epoch_accuracy = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
        
            #Data send to device + requires_grad=True
            images, labels = Variable(images.to(device)), Variable(labels.to(device))

            #Zero the gradient 
            optimizer.zero_grad()
            #Predictions 
            labels_hat = model(images)
            #Loss
            epoch_loss = criterion(labels_hat, labels)
            #Upgrade the gradients (backpropagate) and the optimizer
            epoch_loss.backward()
            optimizer.step()

        #Accuracy
        nof_predictions = 0.0

        #Evaluate the model (different than train to fasten the inference)
        model.eval()

        #Fasten the inference by setting every requires_grad to False
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                #Run the model on the test set
                outputs = model(images)
                #Extract the labels with the maximum probability
                _, labels_hat = torch.max(outputs.data, 1)
                nof_predictions += labels.size(0)
                epoch_accuracy += (labels_hat == labels).sum().item()
        #Compute the accuracy over the test set
        epoch_accuracy = (100*epoch_accuracy/nof_predictions)

        print('Epoch', epoch+1,', test accuracy: {:.4f} % \n'.format(epoch_accuracy))
        
        #Save model when best accuracy is beaten
        if epoch_accuracy > best_accuracy:
            saveModel(file_path_save_model, model)
            best_accuracy = epoch_accuracy

    return model



def plot_inference(model, dataset, device):
    """
        Show the inference
    """

    plt.figure(figsize=(10,10))
    nof_images = len(dataset.data)

    for i in range(12):

        #Select a random image in the dataset
        idx = random.randrange(nof_images)
        #Inference
        label = dataset[idx][1]
        image = dataset[idx][0].unsqueeze(0).to(device)
        #print(image.shape)
        with torch.no_grad():
            model.eval()
            label_hat = torch.max(model(image), 1).indices[0].item()
            #print(torch.max(model(image), 1).indices[0].item())
        #Sent back the image to the CPU
        image = image.squeeze().to('cpu')

        #Plot
        ax = plt.subplot(4,3,i+1)
        ax.set_title("label_hat = {}, label = {}".format(label_hat, label))
        plt.imshow(image, cmap='gray_r')
        plt.axis('off')
    
    plt.show()


plot_inference(LeNet5, mnist_testset)