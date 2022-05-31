"""
Species Classification

0-funestus 1-gambiae 2-Anopheles_other 3-culex 4-All other
"""
#%matplotlib inline
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
import cv2
import re
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import models
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


device='0,1'
if torch.cuda.is_available():
    device = list(map(int,device.split(',')))#torch.device("cuda:0")
    print ("running on the GPU")
else:
    device = torch.device("cpu")
    print ("running on the cpu")

from torch.utils.data import Dataset
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

#Transforms
from torchvision import transforms
invTrans = transforms.Compose([ 
                                transforms.RandomHorizontalFlip(p=0.25),
                                transforms.RandomVerticalFlip(p=0.25),
                                transforms.RandomRotation(degrees=(-45,45), fill=(0,)),
                                transforms.RandomPerspective(distortion_scale=0.2),
                                #transforms.RandomAffine(degrees=(20,80),translate=(0.1,0.3),scale=(0.7,0.9)),
                                transforms.ColorJitter()
                            
                            
                                #transforms.Normalize(mean = [ 0.5150, 0.7373, 1.1327 ],
                                #                     std = [ 0.5055, 0.4621, 0.4949 ]),
    
                            ])
model = models.efficientnet_b1(pretrained=True)
model.fc = nn.Linear(2048, 5)
model = torch.nn.DataParallel(model,device_ids=device).cuda()

EPOCHS = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr = 1e-4, weight_decay = 1e-5)

species_all = ["AF",
            "AG",
            "A_other",
            "Culex",
            "Other"]

def train(train_dataloader,test_dataloader,i):
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    
    
    for epoch in range (EPOCHS+1):
        #print(epoch)
        correct = 0
        total = 0
        train_ave_loss = 0
        model.train()
        for batch_X, batch_Y in train_dataloader:
            
            batch_X = batch_X.cuda()
            batch_Y = batch_Y.cuda()
            #print("CUDA OVERRRRRRRR")
            # zero gradient
            optimizer.zero_grad()
            # pass through
            outputs = model(batch_X)
            # compute loss and back propagate
            loss = criterion(outputs, batch_Y)
            #loss2 = criterion(aux_outputs, batch_Y)
            #loss = loss1 + 0.4 * loss2
            
            loss.backward()
            # optimize
            optimizer.step()
            
            train_ave_loss += loss.data.item()
            _, predicted = outputs.max(1)
            total += batch_Y.size(0)
            correct += predicted.eq(batch_Y).sum().item()
            
        #print("HEREEEEEEEEEEEE")
        train_loss.append(train_ave_loss/len(train_dataloader))
        train_acc.append(100.*correct/total)
        print(f"Epoch: {epoch},Train Loss: {train_ave_loss/len(train_dataloader)} | Train Acc: {100.*correct/total} ({correct}/{total})")
    
        if epoch % 1 == 0:
        
            model.eval()
            valid_correct = 0
            valid_total = 0
            valid_ave_loss = 0
            max_val=0 
            
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
            #print("before")        
            valid_loss.append(valid_ave_loss/len(test_dataloader))
            valid_acc.append(100.*valid_correct/valid_total)
            
            print(f"Validation Loss: {valid_ave_loss/len(test_dataloader)} | Validation Acc: {100.*valid_correct/valid_total} ({valid_correct}/{valid_total})")
            
            if((valid_correct/valid_total)>max_val):
                #print("INSIDE IF")
                max_val=valid_correct/valid_total
                if not os.path.exists('/home/shrutihegde/Desktop/EfficientNet/Species_Model_Test/'):
                    os.makedirs('/home/shrutihegde/Desktop/EfficientNet/Species_Model_Test/')
                torch.save(model.state_dict(),'/home/shrutihegde/Desktop/EfficientNet/Species_Model_Test/TrainModels'+str(i)+'.pt')
    return train_loss, valid_loss, train_acc, valid_acc


def test(test_dataloader):
    correct = 0 
    total = 0
    model.eval()
    out = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = model(inputs)
            #loss = criterion(outputs, test_y[i].to(device).long())
            _, predicted = outputs.max(1)
            out.append(predicted.cpu().detach().numpy())
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    print("Accuracy:", round(correct/total, 3))
    return out

filename = "/home/shrutihegde/Desktop/species/"
for i in range(1,6):
    # train
    print("In loop")

    M1_train_datapath = filename + f"CV_1_M1/train_data_fold{i}.pt"
    M1_train_labelpath = filename + f"CV_1_M1/train_label_fold{i}.pt"
    M2_train_datapath = filename + f"CV_1_M2/train_data_fold{i}.pt"
    M2_train_labelpath = filename + f"CV_1_M2/train_label_fold{i}.pt"

    # M1_train_data = torch.empty(1)
    # M1_train_label = torch.empty(1)
    # M2_train_data = torch.empty(1)
    # M2_train_label = torch.empty(1)
    M1_train_data = torch.load(M1_train_datapath)
    M1_train_label = torch.load(M1_train_labelpath)
    M2_train_data = torch.load(M2_train_datapath)
    M2_train_label = torch.load(M2_train_labelpath)

    # val
    
    M1_val_datapath = filename + f"CV_1_M1/val_data_fold{i}.pt"
    M1_val_labelpath = filename + f"CV_1_M1/val_label_fold{i}.pt"
    M2_val_datapath = filename + f"CV_1_M2/val_data_fold{i}.pt"
    M2_val_labelpath = filename + f"CV_1_M2/val_label_fold{i}.pt"

    # M1_val_data = torch.empty(1)
    # M1_val_label = torch.empty(1)
    # M2_val_data = torch.empty(1)
    # M2_val_label = torch.empty(1)
    M1_val_data = torch.load(M1_val_datapath)
    M1_val_label = torch.load(M1_val_labelpath)
    M2_val_data = torch.load(M2_val_datapath)
    M2_val_label = torch.load(M2_val_labelpath)
    
    # other from old database
    Old_train_data = torch.load(filename + "train_other_data.pt")
    Old_train_label = torch.load(filename + "train_other_label.pt")
    Old_test_data = torch.load(filename + "test_other_data.pt")
    Old_test_label = torch.load(filename + "test_other_label.pt")

    #Concatenate 3 sets - M1, M2, Other
    train_data = torch.empty(1)
    test_data = torch.empty(1)
    train_label = torch.empty(1)
    test_label = torch.empty(1)

    train_data = torch.cat((M1_train_data, M2_train_data, Old_train_data), 0)
    test_data = torch.cat((M1_val_data, M2_val_data, Old_test_data), 0)
    

    train_label = torch.cat((M1_train_label, M2_train_label, Old_train_label), 0)
    test_label = torch.cat((M1_val_label, M2_val_label, Old_test_label), 0)

    

    
    train_dataset=VectorCamDataset(train_data,train_label,transform=invTrans)
    test_dataset=VectorCamDataset(test_data,test_label)

    x, y = train_dataset[1]

    #plt.imshow(torch.permute(x, (1, 2, 0)))

    

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

    train_dataloader = DataLoader(train_dataset,sampler=weighted_sampler, batch_size = 32)
    test_dataloader = DataLoader(test_dataset,batch_size=32, shuffle=False)

    
    
    #model = model.to(device)



    """
    def focal_loss(inputs, targets, reduction="mean"):
        CE_loss = F.cross_entropy(inputs, targets, reduction=reduction)
        pt = torch.exp(-CE_loss)  # prevents nans when probability 0
        F_loss = 0.25 * (1 - pt) ** 5 * CE_loss
        return F_loss.mean()
    """

    
    train_loss, valid_loss, train_acc, valid_acc = train(train_dataloader,test_dataloader,i)

    out = test(test_dataloader)

    out = np.concatenate(out).ravel()

    
    
    # print classification report
    print(classification_report(test_label, out))

    # make confusion matrix
    conf_mat = confusion_matrix(test_label, out)
    conf_mat = conf_mat / np.expand_dims(conf_mat.astype(np.float64).sum(axis=1),1)
    conf_mat = np.round(conf_mat, decimals=2)
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot()
    hm = sn.heatmap(conf_mat, annot=True, ax = ax, cmap="PuBu", fmt='.2', annot_kws={"size": 35 / np.sqrt(len(conf_mat))})
    ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
    ax.set_xlabel('Predicted labels', fontsize = 15)
    ax.set_ylabel('True labels', fontsize = 15)
    ax.xaxis.set_ticklabels(species_all)
    ax.yaxis.set_ticklabels(species_all)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    if not os.path.exists('/home/shrutihegde/Desktop/EfficientNet/Species_Model_Test/'):
        os.makedirs('/home/shrutihegde/Desktop/EfficientNet/Species_Model_Test/')
    plt.savefig('/home/shrutihegde/Desktop/EfficientNet/Species_Model_Test/TrainModels/Species'+str(i)+'.jpg')
    #plt.show();
        



    
