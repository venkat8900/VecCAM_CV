# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
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
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms
from utils import pad, getClassWeights, visualize
from data_prep import specimen_segregator
from train_test_functions import train, test

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the cpu")

## Declare path to the mosquito folder
path = '/home/shrutihegde/Documents/Shruti/Species/Data/cropped/Images_crop/'

## Define Path to store the final csv
store_path = '/home/shrutihegde/Documents/Shruti/Species/Data/cropped/Images_crop/'

#species_all = ["An. funestus","An. gambiae", "An. 'other' ", "Cx.", "Ax. stephensi", "other"]  ## Last class must always be other
#species_all=["AF","AG","A_other", "Culex", "Stephensi", "other"]
species_all=["AF","AG","A_other", "Culex", "other"]
output_path = store_path

train_test_split = 0.8 ## This value can be changed according to requirement

## Obtain list of all image filenames inside the mosquito folder
train_df, test_df = specimen_segregator(path,store_path,".jpg",species_all,train_test_split)
print("Train Test Ratio :", len(train_df)/len(test_df))

class MosquitoDataset(Dataset):
    def __init__(self, data_df, transform=None):
        
        self.count = 0
        self.labels = data_df['label']
        self.transform = transform
        data_df.sample(frac = 1).reset_index(inplace=True, drop=True)
        # shuffle the paths
        self.data_df = data_df

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        
        img_path = self.data_df.loc[idx,'filename']

        image = Image.open(img_path)
        #mage = pad(image,(0,0,0))

        label = self.data_df.loc[idx,'label']
        # label = 0,1,2,3,4 if label==["AF","AG","A_other","Culex","Stephensi"] else 5 if "other"
        self.count+=1
        # print(self.count)
        if self.transform:
            image = self.transform(image)
        
        return image, label


invTrans = transforms.Compose([ 
                                transforms.Resize([300,300]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                transforms.RandomHorizontalFlip(p=0.25),
                                transforms.RandomVerticalFlip(p=0.25),
                                transforms.RandomRotation(degrees=(-45,45), fill=(0,)),
                                transforms.RandomPerspective(distortion_scale=0.2),
                                transforms.ColorJitter()
                            ])

basicTrans = transforms.Compose([ 
                                transforms.Resize([300,300]),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])


ifTrain = True #Train

for i in range(1,6):
    if ifTrain:

        # initialize the model
        #model = models.efficientnet_b1(pretrained=True)
        model = torch.load('/home/shrutihegde/Documents/Shruti/Species/MosquitoNet/code/M0_42_m_d.pt')
       
        EPOCHS = 50
        model.classifier=nn.Linear(1280,len(species_all))
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        # train
        print("In train loop \n")

        train_dataset = MosquitoDataset(train_df, transform=invTrans)
        test_dataset = MosquitoDataset(test_df, transform=basicTrans)


        class_weights_all = getClassWeights(train_dataset)
        #class_weights_all = [1/319, 1/430, 1/547, 1/985, 1/27]

        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
            )

        train_dataloader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=32)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print("Data Prepared : Starting Training \n")

        train_loss, valid_loss, train_acc, valid_acc = train(model, criterion, optimizer, train_dataloader, output_path, EPOCHS, i)

    trained_model = torch.load(output_path+'Train/Model_'+str(i)+'.pt')

    test_dataset = MosquitoDataset(test_df, transform=basicTrans)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_out,_,_ = test(trained_model, test_dataloader)

    test_out = np.concatenate(test_out).ravel()

    outputdf = pd.DataFrame({'test_out':test_out})
    outputdf.to_csv(output_path+'cropped.csv')

    visualize(test_df['label'], test_out, output_path, species_all, i)

