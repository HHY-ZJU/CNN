#!/usr/bin/env python
# coding: utf-8

# Imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import argparse
import setup



data_dir = r'/kaggle/input/dataset/resize_img/resize_img'
label_path = r'/kaggle/input/dataset/resize_img/train_label_110_aug/train_label_110_aug'
checkpoint_path = '/kaggle/working' #results.cp_path
arch = 'resnet50' #results.arch
hidden_units = [1024,1024] #results.hidden_units
epochs = 100 #results.epoch
lr = 0.001 #results.learning_rate
gpu = True #results.boolean_t
print_every = 1
train_loss_list = []
val_loss_list = []

if gpu==True:
    using_gpu = torch.cuda.is_available()
    torch.cuda.set_device(1)
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'
    
    
# Loading Dataset
image_trainloader, image_testloader, image_valloader, image_trainset = setup.loading_data1(data_dir, label_path)

# Network Setup
model, input_size, params = setup.make_model(arch, hidden_units, lr)
criterion = nn.BCELoss()

if(arch == 'vgg16'):
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=.9)
else:
    cnn_params = list(model.transion_layer.parameters()) + list(model.bn_layer.parameters()) + \
                 list(model.relu_layer.parameters()) + list(model.Linear_layer.parameters())
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)


# Training Model
train_loss_list = setup.my_DLM(model, image_trainloader,  epochs, print_every, criterion, optimizer, device,
             train_loss_list)


path1 = '/kaggle/working/train_loss.txt'

f1 = open(path1, 'w')
for each in train_loss_list:
    f1.write(str(each)+'\n')
f1.close()


# Saving Checkpoint
setup.save_checkpoints(model, arch, lr, epochs, input_size, hidden_units,  checkpoint_path)