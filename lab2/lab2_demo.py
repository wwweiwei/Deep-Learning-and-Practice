import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## hyper parameters
Batch_size = 64
Learning_rate = 1
Epochs = 2000

def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

class BCI_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

train_data, train_label, test_data, test_label = read_bci_data()
train_dataset = BCI_dataset(train_data, train_label)
test_dataset = BCI_dataset(test_data, test_label)

train_loader = DataLoader(train_dataset, batch_size = Batch_size)
test_loader = DataLoader(test_dataset, batch_size = Batch_size)

#print("train_loader: "+str(train_loader))
#print("test_loader: "+str(test_loader))

class EEGNet(nn.Module):
    def __init__(self, mode):
        super(EEGNet, self).__init__()
        if mode == 'elu':
            activation = nn.ELU()
        elif mode == 'relu':
            activation = nn.ReLU()
        elif mode == 'leakyrelu':
            activation = nn.LeakyReLU()
        
        ## Conv2D
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        
        ## DepthwiseConv2D
        self.deptwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25))

        ## SeparableConv2D
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25))
        
        ## Classification
        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))
       
    def forward(self, data):
        output = self.firstconv(data)
        output = self.deptwiseConv(output)
        output = self.separableConv(output)
        output = output.view(output.size(0), -1)
        output = self.classify(output)
        
        return output

class DeepConvNet(nn.Module):
    def __init__(self, mode):
        super(DeepConvNet, self).__init__()
        if mode == 'elu':
            activation = nn.ELU()
        elif mode == 'relu':
            activation = nn.ReLU()
        elif mode == 'leakyrelu':
            activation = nn.LeakyReLU()
        self.conv0 = nn.Conv2d(1, 25, kernel_size=(1,5))
        self.conv1 = nn.Sequential(
                nn.Conv2d(25, 25, kernel_size=(2, 1)),
                nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv2 = nn.Sequential(
                nn.Conv2d(25, 50, kernel_size=(1, 5)),
                nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv3 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=(1, 5)),
                nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.conv4 = nn.Sequential(
                nn.Conv2d(100, 200, kernel_size=(1, 5)),
                nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
                activation,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5))
        self.classify = nn.Linear(8600, 2)
        
    def forward(self, data):
        output = self.conv0(data)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(output.size(0), -1)
        output = self.classify(output)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
modes = ['elu', 'relu', 'leakyrelu']

for mode in modes:
    path_model = "./model_EEGNet_"+str(mode)+".pkl"
    model = torch.load(path_model)
    model.eval()
    num_total = 0
    num_corrects = 0
    for data, label in test_loader:
        data = data.to(device).float()
        label = label.to(device).long()
        output = model(data)
        num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
        num_total += len(label)
    accuracy = num_corrects / num_total
    print("[EEGNet] Activation function: ", mode, "=> Accuracy: ", accuracy)
    # print(model)

print("--------------------------")

for mode in modes:
    path_model = "./model_DeepConvNet_"+str(mode)+".pkl"
    model = torch.load(path_model)
    model.eval()
    num_total = 0
    num_corrects = 0
    for data, label in test_loader:
        data = data.to(device).float()
        label = label.to(device).long()
        output = model(data)
        num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
        num_total += len(label)
    accuracy = num_corrects / num_total
    print("[DeepConvNet] Activation function: ", mode, "=> Accuracy: ", accuracy)
    # print(model)
