"""
Abdomen Classification
"UF/FF/SG/G"

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
species_all = ["UF", "FF/SG/G"]
# species_all = ["FF", "SG/G"]

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
            print("Model saved at Epoch: ", epoch)
            if not os.path.exists(args.save_path + 'Saved Models/'):
                os.makedirs(args.save_path + 'Saved Models/')
            torch.save(model, args.save_path + 'Saved Models/' + 'abdomen_fold'+str(i)+'.pt')
    return train_loss, valid_loss, train_acc, valid_acc

def test(test_dataloader, model, threshold=0.5):
    correct = 0 
    total = 0
    model.eval()
    out = []
    valid=[]
    omit=[]

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = F.softmax(model(inputs), dim=1)

            for j in range(len(outputs)):
                p = outputs[j].tolist()
                if max(p) < threshold:
                    valid.append(False)
                    omit.append(0)
                else:
                    valid.append(True)
                    omit.append(1)

            _, predicted = outputs.max(1)
            out.append(predicted.cpu().detach().numpy())
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    print("Accuracy:", round(correct/total, 3))
    return out, np.array(valid), omit


def visualize(true_label, predicted_label, i):
    # print classification report
    
    print(len(true_label))
    print(np.size(predicted_label))
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
    ax.xaxis.set_ticklabels(species_all)
    ax.yaxis.set_ticklabels(species_all)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if not os.path.exists('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Results/Threshold/pt'+str(threshold)+'/'):
            os.makedirs('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Results/Threshold/pt'+str(threshold)+'/')
    plt.savefig('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Results/Threshold/pt'+str(threshold)+'/Model_'+str(i)+'.jpg')



#ifTrain = True #Train
ifTrain=False #Test


if ifTrain:
    for i in range(1, 6):

        # initialize the model
        model = models.efficientnet_b1(pretrained=True)
        #model.fc = nn.Linear(2048, 5)
        model.classifier=nn.Linear(1280,2)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        filename = "/home/shrutihegde/Desktop/species/"

        # train
        print("In train loop")

        M1_train_datapath = filename + f"CV_1_M1/train_data_fold{i}.pt"
        M1_train_labelpath = filename + f"CV_1_M1/train_label_fold{i}.pt"
        M2_train_datapath = filename + f"CV_1_M2/train_data_fold{i}.pt"
        M2_train_labelpath = filename + f"CV_1_M2/train_label_fold{i}.npy"
     
        M1_train_data = torch.load(M1_train_datapath)
        M1_train_label = torch.load(M1_train_labelpath)
        M2_train_data = torch.load(M2_train_datapath)
        M2_train_label = np.load(M2_train_labelpath, allow_pickle=True)
        M2_train_label = M2_train_label.item()
        M2_train_label = torch.tensor(M2_train_label["label"])

        # val

        M1_val_datapath = filename + f"CV_1_M1/val_data_fold{i}.pt"
        M1_val_labelpath = filename + f"CV_1_M1/val_label_fold{i}.pt"
        M2_val_datapath = filename + f"CV_1_M2/val_data_fold{i}.pt"
        M2_val_labelpath = filename + f"CV_1_M2/val_label_fold{i}.npy"

        M1_val_data = torch.load(M1_val_datapath)
        M1_val_label = torch.load(M1_val_labelpath)
        M2_val_data = torch.load(M2_val_datapath)
        M2_val_label = np.load(M2_val_labelpath, allow_pickle=True)
        M2_val_label = M2_val_label.item()
        M2_val_label = torch.tensor(M2_val_label["label"])

        # Concatenate 2 sets - M1, M2

        train_data = torch.cat((M1_train_data, M2_train_data), 0)

        val_data = torch.cat((M1_val_data, M2_val_data), 0)

        train_label = torch.cat((M1_train_label, M2_train_label), 0)
        val_label = torch.cat((M1_val_label, M2_val_label), 0)

        train_dataset = VectorCamDataset(train_data, train_label, transform=invTrans)
        val_dataset = VectorCamDataset(val_data, val_label)

        target_list = torch.tensor(train_dataset.labels)
        class_count = np.array([len(np.where(train_dataset.labels == t)[0]) for t in np.unique(train_dataset.labels)])
        print(class_count)
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        print(class_weights)
        class_weights_all = class_weights[target_list]
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
            )

        train_dataloader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=32)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        train_loss, valid_loss, train_acc, valid_acc = train(train_dataloader, val_dataloader, i)

        #trained_model = torch.load('/home/shrutihegde/Desktop/Species_PCR/Train/Model_'+str(i)+'.pt')

        #val_out = test(val_dataloader, trained_model)

        #val_out = np.concatenate(val_out).ravel()

        #visualize(val_label, val_out, i)
else:
    #test

    filename = "/home/venkat/Venkat/Abdomen Classification/data/abdomen/task1_woculex/"

    for i in range(1, 6):
        trained_model = torch.load('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Saved Models/abdomen_fold'+str(i)+'.pt')

        print('------------------FOLD'+str(i)+'----------------')

        M1_test_datapath = filename + f"CV_1_M1/test_data.pt"
        M1_test_labelpath = filename + f"CV_1_M1/test_label.pt"
        M2_test_datapath = filename + f"CV_1_M2/test_data.pt"
        M2_test_labelpath = filename + f"CV_1_M2/test_label.npy"

        M1_test_data = torch.load(M1_test_datapath)
        M1_test_label = torch.load(M1_test_labelpath)
        M2_test_data = torch.load(M2_test_datapath)
        M2_test_label = np.load(M2_test_labelpath, allow_pickle=True)
        M2_test_label = M2_test_label.item()
        M2_test_label = torch.tensor(M2_test_label["abdomen"])

        test_data = torch.cat((M1_test_data, M2_test_data), 0)
        test_label = torch.cat((M1_test_label, M2_test_label), 0)

        test_dataset = VectorCamDataset(test_data, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        threshold=0.6
        out, valid, omit = test(test_dataloader, trained_model, threshold=threshold)
        out = np.concatenate(out)

        omit_count=0
        label_match=0
        label_mismatch=0
        whole_removed=[[0 for k in range(2)] for l in range(2)] # 5x5 matrix empty

        for j in range(len(omit)):
            if(omit[j]==0):
                if(out[j]==test_label[j]):
                    label_match+=1 
                    whole_removed[out[j]][out[j]]+=1

                else:
                    label_mismatch+=1
                    test_label_np=test_label[j].detach().numpy()
                    whole_removed[out[j]][(test_label_np)]+=1

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()
        hm = sn.heatmap(whole_removed, annot=True, ax=ax, cmap="PuBu")
        ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
        ax.set_xlabel('Predicted labels', fontsize=15)
        ax.set_ylabel('True labels', fontsize=15)
        ax.xaxis.set_ticklabels(species_all)
        ax.yaxis.set_ticklabels(species_all)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if not os.path.exists('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Results/Threshold/pt'+str(threshold)+'/'):
            os.makedirs('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Results/Threshold/pt'+str(threshold)+'/')

        plt.savefig('/home/venkat/Venkat/Abdomen Classification/Save Models/abdomen/task1/Results/Threshold/pt'+str(threshold)+'/Removed_CF'+str(i)+'.jpg')

        #print(whole_removed)
        y_axis = [label_match, label_mismatch]
        x_axis = ["match", "mismatch"]
        removed = np.array([label_match, label_mismatch])
        print('\'true label = predicted label\' count: ',label_match)
        print('\'true label != predicted label\' count: ',label_mismatch)

        omit_count_total = np.size(valid) - np.count_nonzero(valid)
        print("removed count", omit_count_total)
        out = out[valid]
        test_label = np.array(test_label)[valid]
        visualize(test_label, out, i)
