#!/usr/bin/env python
# coding: utf-8

# Imports here
# import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import json
from torch.autograd import Variable
using_gpu = torch.cuda.is_available()
#torch.cuda.set_device(1)
import dataset_processing
import os
import time
#不使用gpu
from sklearn.metrics import matthews_corrcoef


checkpoint_path = '/kaggle/working'

def loading_data1(data_dir, label_path):

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
                                           transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.572, 0.287, 0.225],
                                                                [0.250, 0.185, 0.150])
    ])
    testval_transforms = transforms.Compose([transforms.Resize(224),#256
                                          # transforms.CenterCrop(224),#224
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.572, 0.287, 0.225],
                                                               [0.250, 0.185, 0.150])])


    dset_train = dataset_processing.DatasetProcessing(data_dir,
                                                      label_path, train_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    image_trainloader = torch.utils.data.DataLoader(dset_train, batch_size=32, shuffle=True)

    return image_trainloader,  dset_train
# Build and train your network
# Freeze parameters so we don't backprop through them
from collections import OrderedDict
def make_model(arch, hidden_units, lr):
    output_size = 24
    if arch=="vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088

        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.5)),
            ('fc1', nn.Linear(input_size, hidden_units[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hidden_units[1], output_size)),
            # ('output', nn.LogSoftmax(dim=1))
            ('output', nn.Sigmoid())
        ]))

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = classifier
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        params_to_update = []
        input_size = 2048
        model = Net(model)
    else:
        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.AuxLogits.fc = nn.Linear(768, 24)
        model.fc = nn.Linear(2048,24)
        model.aux_logits = False
        params_to_update=[]
        for name,param in model.named_parameters():
            if param.requires_grad==True:
                params_to_update.append(param)
        input_size = 2048

    return model, input_size, params_to_update

#修改后的resnet网络结构
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

        self.Linear_layer = nn.Linear(2048, 25)

    def forward(self, x):
        x = self.resnet_layer(x)

        x1 = x.view(x.size(0), -1)
        x = self.Linear_layer(x1)

        return x

# Training the model
def my_DLM(model, image_trainloader, epochs, print_every, criterion, optimizer, device, train_loss_list):
    epochs = epochs
    print_every = print_every
    steps = 0

    print (len(image_trainloader))
    # change to cuda
    if device=='gpu':
        model = model.to('cuda')
    N_count = 0
    for e in range(epochs):
        running_loss = 0
        for batch_idx, (inputs, labels) in enumerate(image_trainloader):

            #print (labels.shape[0])
            size = len(image_trainloader)
            steps += 1
            N_count += inputs.size(0)

            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = F.sigmoid(model.forward(inputs))
            # loss = criterion(outputs, labels)
            xx = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels.float())#/labels.shape[0]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #print(loss.item())
            train_loss_list.append(loss.item())


            # print("Epoch: {}/{}... | ".format(e+1, epochs),
            #           "Loss: {:.4f} | ".format(running_loss/print_every),
            #           )
            if (batch_idx + 1) % print_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e + 1, N_count, len(image_trainloader.dataset), 100. * (batch_idx + 1) / len(image_trainloader),
                    loss.item()))

            running_loss = 0

        # save Pytorch models of best record
        if((e + 1) % 10 == 0):
            torch.save(model.state_dict(), os.path.join(checkpoint_path, 'resnet50_epoch{}.pth'.format(e + 1)))  # save spatial_encoder
            print("Epoch {} model saved!".format(e + 1))
    return train_loss_list

# Do validation on the test set
def testing(model, dataloader):
    model.eval()
    model.to('cuda')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = F.sigmoid(model(inputs))
            _ , prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels.data).sum().item()
        print('Accuracy on the test set: %d %%' % (100 * correct / total))   


# Save the checkpoint 
def save_checkpoints(model, arch, lr, epochs, input_size, hidden_units,  checkpoint_path):
    # model.class_to_idx = class_to_idx
    state = {
            'structure' :arch,
            'learning_rate': lr,
            'epochs': epochs,
            'input_size': input_size,
            'hidden_units':hidden_units,
            'state_dict':model.state_dict(),
            # 'class_to_idx': model.class_to_idx
        }
    torch.save(state, checkpoint_path + 'command_checkpoint_resnet50-2.pth')
    print('Checkpoint saved in ', checkpoint_path + 'command_checkpoint.pth')

    

# Write a function that loads a checkpoint and rebuilds the model
def loading_checkpoint(path):
    
    # Loading the parameters
    state = torch.load(path)
    lr = state['learning_rate']
    input_size = state['input_size']
    structure = state['structure']
    hidden_units = state['hidden_units']
    epochs = state['epochs']
    
    # Building the model from checkpoints
    model,_ = make_model(structure, hidden_units, lr)
    #model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    return model


# Inference for classification
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img

# Labeling
def labeling(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

idx_to_class = {'0': '食管', '1': '齿状线', '2': '贲门', '3': '胃底', '4': '胃体大弯上部', '5': '胃体大弯中部', '6': '胃体大弯下部',
                            '7': '胃体后壁上部', '8': '胃体后壁中部', '9': '胃体后壁下部',
                            '10': '胃体前壁上部', '11': '胃体前壁中部', '12': '胃体前壁下部', '13': '胃体小弯上部', '14': '胃体小弯中部',
                            '15': '胃体小弯下部', '16': '胃窦大弯', '17': '胃窦后壁', '18': '胃窦前壁',
                            '19': '胃窦小弯', '20': '胃角', '21': '幽门', '22': '十二指肠球部', '23': '十二指肠降部'}
# Class Prediction
def predict(processed_image, model, topk, device,result,fea):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.eval()
    model.to('cuda')
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.to('cuda')
    txtpath = './prelabel.txt'
    with torch.no_grad():
        output = F.sigmoid(model.forward(processed_image))
        # probs, classes = torch.topk(input=output, k=topk)
        # top_prob = probs.exp()
        #print(output)

    op = []
        # import pdb
        # pdb.set_trace()
    for each in output.cpu().numpy():
        print(each)
        nb = 0

        index = []
        for each0 in each:
            if each0 >= 0.5:
                op.append(1)
                index.append(nb)
            else:
                op.append(0)
            nb = nb + 1

    fea.append(op)

    top_classes = [idx_to_class[str(each)] for each in index]
    for each in top_classes:
        if (each not in result):
            result.append(each)
    #print(top_classes)




    # Convert indices to classes
    # idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    # top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return result,fea ,op