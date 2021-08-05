import pandas as pd
from torch.utils import data
import numpy as np
import torch

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms

from sklearn.metrics import confusion_matrix
from collections import Counter
from PIL import Image
import gc
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 10

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

        ## transform
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        ## step 1
        image_path = './data/' + self.img_name[index] + '.jpeg'
        ## step 2
        label = self.label[index]
        ## step 3
        img = Image.open(image_path).convert('RGB')
        
        if self.mode == 'train' and self.label[index] == 0:
            img = self.transform_train(img).numpy()
        else:
            img = self.transform(img).numpy()

        return img, label

train_dataset = RetinopathyLoader('', 'train')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = RetinopathyLoader('', 'test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def accuracy_comparison(model, epochs, w_train_accuracy, w_test_accuracy, wo_train_accuracy, wo_test_accuracy):
    w_train, = plt.plot(epochs, w_train_accuracy)
    w_test, = plt.plot(epochs, w_test_accuracy)
    wo_train, = plt.plot(epochs, wo_train_accuracy)
    wo_test, = plt.plot(epochs, wo_test_accuracy)

    plt.title('Result Comparison(' + model + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend([w_train, w_test, wo_train, wo_test], ['Train(with pretraining)', 'Test(with pretraining)', 'Train(w/o pretraining)', 'Test(w/o pretraining)'], loc = 'upper left')
    plt.savefig(model + '.jpg', dpi = 300, transparent = True)
    
def draw_confusion_matrix(model, mode, labels, preds):
    fig, ax = plt.subplots(figsize=(7, 5))
    tmp_matrix = confusion_matrix(labels, preds)
    c_matrix = np.zeros((len(tmp_matrix[0]), len(tmp_matrix[0])))

    for i in range(len(tmp_matrix)):
        for j in range(len(tmp_matrix[i])):
            c_matrix[i][j] = tmp_matrix[i][j] / sum(tmp_matrix[i])

    sn.heatmap(c_matrix, annot = True, annot_kws = {'size': 10}, cmap = 'GnBu', fmt = '.2f', ax = ax, square = True)
    ax.set_title('Normalized confusion matrix')
    ax.xaxis.set_tick_params(rotation = 0)
    ax.yaxis.set_tick_params(rotation = 0)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.title.set_fontsize(16)
    plt.savefig(model + '_' + mode +'.jpg', dpi = 300, transparent = True)

class BasicBlock(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dimensions, out_channels=out_dimensions, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dimensions, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_dimensions, out_channels=out_dimensions, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dimensions, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            BasicBlock(in_dimensions=64, out_dimensions=64, stride=(1, 1)),
            BasicBlock(in_dimensions=64, out_dimensions=64, stride=(1, 1)))

        self.layer2 = nn.Sequential(
            BasicBlock(in_dimensions=64, out_dimensions=128, stride=(2, 2), downsample=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), bias=False),
               nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            BasicBlock(in_dimensions=128, out_dimensions=128, stride=(1, 1)))

        self.layer3 = nn.Sequential(
            BasicBlock(in_dimensions=128, out_dimensions=256, stride=(2, 2), downsample=nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), bias=False),
               nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            BasicBlock(in_dimensions=256, out_dimensions=256, stride=(1, 1)))
        self.layer4 = nn.Sequential(
            BasicBlock(in_dimensions=256, out_dimensions=512, stride=(2, 2), downsample=nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), bias=False),
               nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            BasicBlock(in_dimensions=512, out_dimensions=512, stride=(1, 1)))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=1000, bias=True)
        self.relu = nn.ReLU()
        self.out = nn.Linear(in_features=1000, out_features=5, bias=True)

    def forward(self, img):
        output = self.conv1(img)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.relu(output)
        output = self.out(output)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

modes = ['with_pretraining', 'without_pretraining']

## ResNet18
for mode in modes:
    model = ResNet18()
    if mode == 'with_pretraining':
        pretraining_model = models.resnet18(pretrained=True)
        weights = model.state_dict()
        weights.update(pretraining_model.state_dict())
        model.load_state_dict(weights)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
        )

    model.to(device)
    ce_loss.to(device)
    gc.collect()

    if mode == 'with_pretraining':
        w_train_accuracy = []
        w_test_accuracy = []
    elif mode == 'without_pretraining':
        wo_train_accuracy = []
        wo_test_accuracy = []

    epochs = []
    best_accuracy = 0
    for epoch in range(EPOCHS):
        epochs.append(epoch + 1)
        model.train()
        num_total = 0
        num_corrects = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device).float()
            label = label.to(device).long()

            output = model(img)
            loss = ce_loss(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if mode == 'with_pretraining':
            w_train_accuracy.append(accuracy)
        elif mode == 'without_pretraining':
            wo_train_accuracy.append(accuracy)

        gc.collect()

        model.eval()
        num_total = 0
        num_corrects = 0
        for img, label in test_loader:
            img = img.to(device).float()
            label = label.to(device).long()

            output = model(img)
            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        accuracy = num_corrects / num_total
        if mode == 'with_pretraining':
            w_test_accuracy.append(accuracy)
        elif mode == 'without_pretraining':
            wo_test_accuracy.append(accuracy)

    print(mode, best_accuracy)
    torch.save(model.state_dict(), mode + '_resnet18.pt')

accuracy_comparison('ResNet18', epochs, w_train_accuracy, w_test_accuracy, wo_train_accuracy, wo_test_accuracy)

for mode in modes:
    model = ResNet18()
    model.load_state_dict(torch.load(mode + '_resnet18.pt'))
    model.to(device)
    model.eval()
    preds = []
    labels = []
    for img, label in test_loader:
        img = img.to(device).float()
        output = model(img)

        preds.extend(list((torch.argmax(output, dim=1)).detach().cpu().numpy()))
        labels.extend(list(label))

    draw_confusion_matrix('ResNet18', mode, labels, preds)

## ResNet50
class Bottleneck(nn.Module):
    def __init__(self, in_dimensions, hidden_dimensions, out_dimensions, stride, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_dimensions, out_channels=hidden_dimensions, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dimensions, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dimensions, out_channels=hidden_dimensions, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dimensions, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(in_channels=hidden_dimensions, out_channels=out_dimensions, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_dimensions, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            Bottleneck(in_dimensions=64, hidden_dimensions=64, out_dimensions=256, stride=(1, 1), downsample=nn.Sequential(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), bias=False),
               nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            Bottleneck(in_dimensions=256, hidden_dimensions=64, out_dimensions=256, stride=(1, 1)),
            Bottleneck(in_dimensions=256, hidden_dimensions=64, out_dimensions=256, stride=(1, 1)))

        self.layer2 = nn.Sequential(
            Bottleneck(in_dimensions=256, hidden_dimensions=128, out_dimensions=512, stride=(2, 2), downsample=nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), bias=False),
               nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            Bottleneck(in_dimensions=512, hidden_dimensions=128, out_dimensions=512, stride=(1, 1)),
            Bottleneck(in_dimensions=512, hidden_dimensions=128, out_dimensions=512, stride=(1, 1)),
            Bottleneck(in_dimensions=512, hidden_dimensions=128, out_dimensions=512, stride=(1, 1)))

        self.layer3 = nn.Sequential(
           Bottleneck(in_dimensions=512, hidden_dimensions=256, out_dimensions=1024, stride=(2, 2), downsample=nn.Sequential(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
               nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            Bottleneck(in_dimensions=1024, hidden_dimensions=256, out_dimensions=1024, stride=(1, 1)),
            Bottleneck(in_dimensions=1024, hidden_dimensions=256, out_dimensions=1024, stride=(1, 1)),
            Bottleneck(in_dimensions=1024, hidden_dimensions=256, out_dimensions=1024, stride=(1, 1)),
            Bottleneck(in_dimensions=1024, hidden_dimensions=256, out_dimensions=1024, stride=(1, 1)),
            Bottleneck(in_dimensions=1024, hidden_dimensions=256, out_dimensions=1024, stride=(1, 1)))

        self.layer4 = nn.Sequential(
            Bottleneck(in_dimensions=1024, hidden_dimensions=512, out_dimensions=2048, stride=(2, 2), downsample=nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
               nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))),
            Bottleneck(in_dimensions=2048, hidden_dimensions=512, out_dimensions=2048, stride=(1, 1)),
            Bottleneck(in_dimensions=2048, hidden_dimensions=512, out_dimensions=2048, stride=(1, 1)))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.out = nn.Linear(in_features=1000, out_features=5, bias=True)

    def forward(self, img):
        output = self.conv1(img)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = self.out(output)
        return output

for mode in modes:
    model = ResNet50()
    if mode == 'with_pretraining':
        pretraining_model = models.resnet50(pretrained=True)
        weights = model.state_dict()
        weights.update(pretraining_model.state_dict())
        model.load_state_dict(weights)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
        )

    model.to(device)
    ce_loss.to(device)
    gc.collect()

    if mode == 'with_pretraining':
        w_train_accuracy = []
        w_test_accuracy = []
    elif mode == 'without_pretraining':
        wo_train_accuracy = []
        wo_test_accuracy = []

    epochs = []
    best_accuracy = 0
    for epoch in range(EPOCHS):
        epochs.append(epoch + 1)
        model.train()
        num_total = 0
        num_corrects = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device).float()
            label = label.to(device).long()

            output = model(img)
            loss = ce_loss(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if mode == 'with_pretraining':
            w_train_accuracy.append(accuracy)
        elif mode == 'without_pretraining':
            wo_train_accuracy.append(accuracy)

        gc.collect()

        model.eval()
        num_total = 0
        num_corrects = 0
        for img, label in test_loader:
            img = img.to(device).float()
            label = label.to(device).long()

            output = model(img)
            num_corrects += (torch.argmax(output, dim=1) == label).sum().item()
            num_total += len(label)
        accuracy = num_corrects / num_total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        accuracy = num_corrects / num_total
        if mode == 'with_pretraining':
            w_test_accuracy.append(accuracy)
        elif mode == 'without_pretraining':
            wo_test_accuracy.append(accuracy)

    print(mode, best_accuracy)
    torch.save(model.state_dict(), mode + '_resnet50.pt')

accuracy_comparison('ResNet50', epochs, w_train_accuracy, w_test_accuracy, wo_train_accuracy, wo_test_accuracy)

modes = ['with_pretraining', 'without_pretraining']
for mode in modes:
    model = ResNet50()
    model.load_state_dict(torch.load(mode + '_resnet50.pt'))
    model.to(device)
    model.eval()
    preds = []
    labels = []
    for img, label in test_loader:
        img = img.to(device).float()
        output = model(img)

        preds.extend(list((torch.argmax(output, dim=1)).detach().cpu().numpy()))
        labels.extend(list(label))

    draw_confusion_matrix('ResNet50', mode, labels, preds)
