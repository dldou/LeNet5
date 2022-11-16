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

def saveModel(model, file_path):
    """
        Function to save model's parameters
    """
    torch.save(model.state_dict(), file_path)


def loadModel(model, file_path, device):
    """
        Function to load function when only the params have been saved
    """
    params = torch.load(file_path)
    model.load_state_dict(params)


def checkPoint_model(model, 
                     optimizer, loss, epoch,
                     file_path):
    """
        Function to save model's checkpoints
    """
    
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                file_path)

def load_checkPoint_model(model, optimizer, file_path, device):

    checkpoint = torch.load(file_path)

    #Loading
    model.load_state_dict(checkpoint['model_state_dict'], map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=device)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, loss


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
            saveModel(model, file_path_save_model)
            best_accuracy = epoch_accuracy

    return model


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
            saveModel(model, file_path_save_model)
            best_accuracy = epoch_accuracy

    return model


def plot_inference(model, dataset):

    fig = plt.figure(figsize=(10,10))
    nof_images = len(dataset.data)

    for i in range(20):

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
        ax = plt.subplot(5,4,i+1)
        ax.set_title("predicted label = {}".format(label_hat))
        plt.imshow(image, cmap='gray_r')
        plt.axis('off')
    
    fig.suptitle("{}'s predictions on few examples".format(model.__class__.__name__), fontsize=16)

    #Save
    plt.savefig(str(model.__class__.__name__) + "_accuraccy_97.1" + ".pdf")

    #Show
    plt.show()
    
 
    
def displayConvFilers(model, 
                      layer_name,
                      optimizer, epoch,
                      figsize=(2,2),
                      ):

    layer = model.state_dict()[layer_name]
    batch, channels, height, width = layer.shape
    nof_filters = batch*channels

    fig = plt.figure(figsize=figsize)

    for b_idx in range(batch):
        for c_idx in range(channels):

            #Get the filter on the batch n°b_idx on the channel n°c_idx
            filter = layer[b_idx][c_idx].cpu()
            #Filter is number i in the subplot
            i = b_idx*channels + c_idx

            ax = plt.subplot(batch, channels, i+1)
            plt.imshow(filter, cmap='gray')

            #Layout of the subplots
            #Withdraw ticks
            ax.set_yticks([])
            ax.set_xticks([])
            if (c_idx==0):
                ax.set_ylabel("Batch #{}".format(b_idx+1), fontsize=20)
            #Display the number of channel at the bottom of the figure
            if b_idx == (batch - 1):
                ax.set_xlabel("Channel #{}".format(c_idx+1), fontsize=20)

    #Layout of the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.suptitle("Visualization of layer's filters on {} model (unpruned)\n".format(model.__class__.__name__) 
              + "(Model characteristics - optimizer: {}, learning rate: {}, number of epochs: {})\n \n".format(optimizer.__class__.__name__, 
                                                                                                               optimizer.state_dict()['param_groups'][0]['lr'],
                                                                                                               str(epoch)) 
              + "Layer name: {}, Filters' size: ({}x{})".format(layer_name, 
                                                                height, width)
             , fontsize=25)

    #Save figure
    plt.savefig(str(model.__class__.__name__) + "_" + layer_name + "_" 
                + "batch_" + str(batch) + "_" 
                + "channels_" + str(channels)  + "_" 
                + "height" + str(height) + "_" 
                + "width_" + str(width)  
                + ".pdf"
                )
    
    #Show figure
    plt.show()