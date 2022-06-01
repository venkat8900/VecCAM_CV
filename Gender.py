"""
Gender Classification

0-female 1-male
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import models
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the cpu")

class VectorCamDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        'Initialization'
        self.labels = labels
        self.data = data
        self.transform = transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)
    def __getitem__(self, index):
        
        'Generates one sample of data'
        # Load data and get label
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y

# Transforms
from torchvision import transforms
invTrans = transforms.Compose([ 
                                transforms.RandomHorizontalFlip(p=0.25),
                                transforms.RandomVerticalFlip(p=0.25),
                                transforms.RandomRotation(degrees=(-45,45), fill=(0,)),
                                transforms.RandomPerspective(distortion_scale=0.2),
                                transforms.ColorJitter()
    
                            ])

EPOCHS = 50
gender_all = ["female", "male"]

def train(train_dataloader,test_dataloader,i):
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    max_val = 0

    for epoch in range(EPOCHS+1):

        correct = 0
        total = 0
        train_ave_loss = 0
        model.train()
        for batch_X, batch_Y in train_dataloader:
            
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()

            # zero gradient
            optimizer.zero_grad()
            # pass through
            outputs = model(batch_X)
            # compute loss and back propagate
            loss = criterion(outputs, batch_Y)
            
            loss.backward()
            # optimize
            optimizer.step()
            
            train_ave_loss += loss.data.item()
            _, predicted = outputs.max(1)
            total += batch_Y.size(0)
            correct += predicted.eq(batch_Y).sum().item()

        train_loss.append(train_ave_loss/len(train_dataloader))
        train_acc.append(100.*correct/total)
        print(f"Epoch: {epoch},Train Loss: {train_ave_loss/len(train_dataloader)} | Train Acc: {100.*correct/total} ({correct}/{total})")
    
        # ======  Validation ======
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_ave_loss = 0

        with torch.no_grad():
            for valid_batch_X, valid_batch_Y in test_dataloader:

                valid_batch_X = valid_batch_X.cuda()
                valid_batch_Y = valid_batch_Y.cuda()

                valid_outputs = model(valid_batch_X)
                loss = criterion(valid_outputs, valid_batch_Y)
                valid_ave_loss += loss.data.item()

                _, predicted = valid_outputs.max(1)
                valid_correct += predicted.eq(valid_batch_Y).sum().item()
                valid_total += valid_batch_Y.size(0)

        valid_loss.append(valid_ave_loss/len(test_dataloader))
        valid_acc.append(100.*valid_correct/valid_total)

        print(f"Validation Loss: {valid_ave_loss/len(test_dataloader)} | Validation Acc: {100.*valid_correct/valid_total} ({valid_correct}/{valid_total})")

        if (valid_correct / valid_total) > max_val:
            max_val = valid_correct/valid_total
            print("saved")
            torch.save(model, '../model/gender/gender_fold'+str(i)+'.pt')
    return train_loss, valid_loss, train_acc, valid_acc


def test(test_dataloader, model):
    correct = 0 
    total = 0
    model.eval()
    out = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            out.append(predicted.cpu().detach().numpy())
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    print("Accuracy:", round(correct/total, 3))
    return out


def visualize(true_label, predicted_label):
    # print classification report
    print(classification_report(true_label, predicted_label))

    # make confusion matrix
    conf_mat = confusion_matrix(true_label, predicted_label)
    conf_mat = conf_mat / np.expand_dims(conf_mat.astype(np.float64).sum(axis=1), 1)
    conf_mat = np.round(conf_mat, decimals=2)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    hm = sn.heatmap(conf_mat, annot=True, ax=ax, cmap="PuBu", fmt='.2', annot_kws={"size": 35 / np.sqrt(len(conf_mat))})
    ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.xaxis.set_ticklabels(gender_all)
    ax.yaxis.set_ticklabels(gender_all)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
    # plt.savefig('../model/gender/gender_fold' + str(i) + '.png')


# iftrain = True
# if iftrain:
#     for i in range(1, 6):
#
#         # initialize the model
#         model = models.efficientnet_b1(pretrained=True)
#         model.fc = nn.Linear(2048, 2)
#         model = model.to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#
#         # train
#         print("In loop")
#
#         M1_train_data_path = f"../data/gender/CV_1_M1/train_data_fold{i}.pt"
#         M1_train_label_path = f"../data/gender/CV_1_M1/train_label_fold{i}.pt"
#         M2_train_data_path = f"../data/gender/CV_1_M2/train_data_fold{i}.pt"
#         M2_train_label_path = f"../data/gender/CV_1_M2/train_label_fold{i}.npy"
#
#         M1_train_data = torch.load(M1_train_data_path)
#         M1_train_label = torch.load(M1_train_label_path)
#         M2_train_data = torch.load(M2_train_data_path)
#         M2_train_label = np.load(M2_train_label_path, allow_pickle=True)
#         M2_train_label = M2_train_label.item()
#         M2_train_label = torch.tensor(M2_train_label["gender"])
#
#         # val
#
#         M1_val_data_path = f"../data/gender/CV_1_M1/val_data_fold{i}.pt"
#         M1_val_label_path = f"../data/gender/CV_1_M1/val_label_fold{i}.pt"
#         M2_val_data_path = f"../data/gender/CV_1_M2/val_data_fold{i}.pt"
#         M2_val_label_path = f"../data/gender/CV_1_M2/val_label_fold{i}.npy"
#
#         M1_val_data = torch.load(M1_val_data_path)
#         M1_val_label = torch.load(M1_val_label_path)
#         M2_val_data = torch.load(M2_val_data_path)
#         M2_val_label = np.load(M2_val_label_path, allow_pickle=True)
#         M2_val_label = M2_val_label.item()
#         M2_val_label = torch.tensor(M2_val_label["gender"])
#
#         # Concatenate 2 sets - M1, M2
#
#         train_data = torch.cat((M1_train_data, M2_train_data), 0)
#         test_data = torch.cat((M1_val_data, M2_val_data), 0)
#
#         train_label = torch.cat((M1_train_label, M2_train_label), 0)
#         test_label = torch.cat((M1_val_label, M2_val_label), 0)
#
#         train_dataset = VectorCamDataset(train_data, train_label, transform=invTrans)
#         test_dataset = VectorCamDataset(test_data, test_label)
#
#         target_list = torch.tensor(train_dataset.labels)
#         class_count = np.array([len(np.where(train_dataset.labels == t)[0]) for t in np.unique(train_dataset.labels)])
#         print(class_count)
#         class_weights = 1./torch.tensor(class_count, dtype=torch.float)
#         print(class_weights)
#         class_weights_all = class_weights[target_list]
#         weighted_sampler = WeightedRandomSampler(
#             weights=class_weights_all,
#             num_samples=len(class_weights_all),
#             replacement=True
#             )
#
#         train_dataloader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=32)
#         test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#         # train_loss, valid_loss, train_acc, valid_acc = train(train_dataloader, test_dataloader, i)
#
#         test_model = torch.load('../model/gender/gender_fold'+str(i)+'.pt')
#
#         out = test(test_dataloader, test_model)
#
#         out = np.concatenate(out).ravel()
#
#         # np.save(f"../model/gender/val_predicted_label_fold{i}.npy", out)
#         # np.save(f"../model/gender/val_true_label_fold{i}.npy", test_label)
#
#     # visualize(test_label, out)
# else:
#     # test
#     test_model = torch.load('../model/gender/gender_fold4.pt')
#
#     M1_test_data_path = f"../data/gender/CV_1_M1/test_data.pt"
#     M1_test_label_path = f"../data/gender/CV_1_M1/test_label.pt"
#     M2_test_data_path = f"../data/gender/CV_1_M2/test_data.pt"
#     M2_test_label_path = f"../data/gender/CV_1_M2/test_label.npy"
#
#     M1_test_data = torch.load(M1_test_data_path)
#     M1_test_label = torch.load(M1_test_label_path)
#     M2_test_data = torch.load(M2_test_data_path)
#     M2_test_label = np.load(M2_test_label_path, allow_pickle=True)
#     M2_test_label = M2_test_label.item()
#     M2_test_label = torch.tensor(M2_test_label["gender"])
#
#     test_data = torch.cat((M1_test_data, M2_test_data), 0)
#     test_label = torch.cat((M1_test_label, M2_test_label), 0)
#
#     test_dataset = VectorCamDataset(test_data, test_label)
#     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#     out = test(test_dataloader, test_model)
#     out = np.concatenate(out).ravel()
#
#     visualize(test_label, out)

true1 = np.load(f"../model/gender/val_true_label_fold1.npy")
predict1 = np.load(f"../model/gender/val_predicted_label_fold1.npy")
true2 = np.load(f"../model/gender/val_true_label_fold2.npy")
predict2 = np.load(f"../model/gender/val_predicted_label_fold2.npy")
true3 = np.load(f"../model/gender/val_true_label_fold3.npy")
predict3 = np.load(f"../model/gender/val_predicted_label_fold3.npy")
true4 = np.load(f"../model/gender/val_true_label_fold4.npy")
predict4 = np.load(f"../model/gender/val_predicted_label_fold4.npy")
true5 = np.load(f"../model/gender/val_true_label_fold5.npy")
predict5 = np.load(f"../model/gender/val_predicted_label_fold5.npy")

true = np.concatenate([true1, true2, true3, true4, true5])
predict = np.concatenate([predict1, predict2, predict3, predict4, predict5])
visualize(true, predict)



    
