"""
Abdomen Classification

python abdomen.py --epochs 50 --data_path "./data/abdomen/task1_woculex/" --categories ["UF", "FF/SG/G"] --train True --test_model_path "./Save Models/abdomen/task1/" --save_path "./Save Models/abdomen/task1/"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from collections import Counter
import argparse
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, help = 'Number of epochs to train th model', default = 50)
parser.add_argument('--weight_decay', type = float, help = 'weight Deacy', default = 0.00001)
parser.add_argument('--lr', type = int, help = 'Learning Rate', default = 0.0001)
parser.add_argument('--data_path', type = str, help = 'path to train and test data', required = True)
parser.add_argument('--device_ids', type = str, default = '0,1', help = 'For example 0,1 to run on two GPUs')
# parser.add_argument('--categories', type = str, help = 'list of classes')
parser.add_argument('--majority', type = bool, default = False, help = 'True if you want to use majority voting criteria')
parser.add_argument('--train', type = bool, default = True, help = 'True if you want to train the model')
parser.add_argument('--test_model_path', type = str, help = 'path to saved model to load during testing', required = True)
parser.add_argument('--save_path', type = str, help = 'path to save results', required = True)
parser.add_argument('--test', type = bool, default = True, help = 'True if you want to test the model')
parser.add_argument('--threshold', type = float, help = 'threshold cutoff', default = 0.7)
parser.add_argument('--fold', type = int, help = "Number of folds", default = 5)


args = parser.parse_args()


device = args.device_ids
# device = '0,1'
if torch.cuda.is_available():
    device = list(map(int,device.split(',')))
    print ("running on the GPU")
else:
    device = torch.device("cpu")
    print ("running on the cpu")


class VectorCamDataset(Dataset):
    """
    Dataloader 
    """

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
                                # transforms.RandomPerspective(distortion_scale=0.2),
                                # transforms.ColorJitter()
    
                            ])

EPOCHS = args.epochs
# abdomen_categories = args.categories
abdomen_categories = ["UF", "FF/SG/G"]
# abdomen_categories = ["FF", "SG/G"]

def train(train_dataloader,test_dataloader,i):
    """
    Function to train the model
    : train_dataloader: train_dataloader
    : test_dataloader: test_dataloader
    : i: fold
    """

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

        if ((valid_correct / valid_total) > max_val) and epoch>10:
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
    valid = []
    omit = []

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


def visualize(true_label, predicted_label, i, fold=None):
    """
    Prints classification report and saves confusion matrix
    : true_label: true label in the test set
    : predicted label: predicted label on the test set by the model
    """
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
    ax.xaxis.set_ticklabels(abdomen_categories)
    ax.yaxis.set_ticklabels(abdomen_categories)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('abdomen_test_majority')
    if not os.path.exists(args.save_path + 'Results/Plots/'):
        os.makedirs(args.save_path + 'Results/Plots/')
    plt.savefig(args.save_path + 'Results/Plots/Confusion_Matrix_Fold_' +str(i)+ '.jpg')


def plot_graph(train_acc, val_acc, train_loss, val_loss):
    """
    Plots graphs of train, validation accuracy and loss
    : train_acc: 1D array of training accuracy for every epoch
    : val_acc: 1D array of validation accuracy for every epoch
    : train_loss: 1D array of training loss for every epoch
    : val_loss: 1D array of validation accuracy for every epoch
    """

    # plot figures for train and val loss. 
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(valid_loss,label="val")
    plt.plot(train_loss,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists(args.save_path + 'Results/Plots/'):
        os.makedirs(args.save_path + 'Results/Plots/')
    plt.savefig(args.save_path + 'Results/Plots/Abdomen'+str(i)+'Train_Val_loss.jpg')

    # plot figures for train and val acc
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Accuracy")
    plt.plot(valid_acc,label="val")
    plt.plot(train_acc,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    if not os.path.exists(args.save_path + 'Results/Plots/'):
        os.makedirs(args.save_path + 'Results/Plots/')
    plt.savefig(args.save_path + 'Results/Plots/Abdomen_Fold_'+str(i)+'_Train_Val_accuracy.jpg')
    



# iftrain = args.train
iftrain= True
iftest = args.test
iftest = False
majority = args.majority
if iftrain:
    for i in range(1, args.fold + 1):

        # initialize the model
        model = models.efficientnet_b1(pretrained=True)
        # print(model)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 2),
        )
        model = torch.nn.DataParallel(model,device_ids=device).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # train
        print("In loop")

        M1_train_data_path = args.data_path + f"CV_1_M1/train_data_fold{i}.pt"
        M1_train_label_path = args.data_path + f"CV_1_M1/train_label_fold{i}.pt"
        M2_train_data_path = args.data_path + f"CV_1_M2/train_data_fold{i}.pt"
        M2_train_label_path = args.data_path + f"CV_1_M2/train_label_fold{i}.npy"

        M1_train_data = torch.load(M1_train_data_path)
        M1_train_label = torch.load(M1_train_label_path)
        M2_train_data = torch.load(M2_train_data_path)
        # M2_train_label = torch.load(M2_train_label_path)
        M2_train_label = np.load(M2_train_label_path, allow_pickle=True)
        M2_train_label = M2_train_label.item()
        M2_train_label = torch.tensor(M2_train_label["abdomen"])

        # val

        M1_val_data_path = args.data_path + f"CV_1_M1/val_data_fold{i}.pt"
        M1_val_label_path =  args.data_path + f"/CV_1_M1/val_label_fold{i}.pt"
        M2_val_data_path =  args.data_path + f"CV_1_M2/val_data_fold{i}.pt"
        M2_val_label_path =  args.data_path + f"CV_1_M2/val_label_fold{i}.npy"

        M1_val_data = torch.load(M1_val_data_path)
        M1_val_label = torch.load(M1_val_label_path)
        M2_val_data = torch.load(M2_val_data_path)
        # M2_val_label = torch.load(M2_val_label_path)
        M2_val_label = np.load(M2_val_label_path, allow_pickle=True)
        M2_val_label = M2_val_label.item() 
        M2_val_label = torch.tensor(M2_val_label["abdomen"])

        # Concatenate 2 sets - M1, M2

        train_data = torch.cat((M1_train_data, M2_train_data), 0)
        val_data = torch.cat((M1_val_data, M2_val_data), 0)

        train_label = torch.cat((M1_train_label, M2_train_label), 0)
        val_label = torch.cat((M1_val_label, M2_val_label), 0)

        train_dataset = VectorCamDataset(train_data, train_label, transform=invTrans)
        val_dataset = VectorCamDataset(val_data, val_label)
        # test
        M1_test_data_path = args.data_path + f"CV_1_M1/test_data.pt"
        M1_test_label_path = args.data_path + f"CV_1_M1/test_label.pt"
        M2_test_data_path = args.data_path + f"CV_1_M2/test_data.pt"
        M2_test_label_path = args.data_path + f"CV_1_M2/test_label.npy"

        M1_test_data = torch.load(M1_test_data_path)
        M1_test_label = torch.load(M1_test_label_path)
        M2_test_data = torch.load(M2_test_data_path)
        # M2_test_label = torch.load(M2_test_label_path)
        M2_test_label = np.load(M2_test_label_path, allow_pickle=True)
        #print()
        M2_test_label = M2_test_label.item()
        M2_test_label = torch.tensor(M2_test_label["abdomen"])

        train_data = torch.empty(1)
        test_data = torch.empty(1)
        train_label = torch.empty(1)
        test_label = torch.empty(1)

        test_data = torch.cat((M1_test_data, M2_test_data), 0)
        test_label = torch.cat((M1_test_label, M2_test_label), 0)
        print("DATA SIZEEEE",test_data.size())
        print("LABEL SIZEEEE",test_label.size())

        test_dataset = VectorCamDataset(test_data, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

        test_model = torch.load(args.save_path + 'Saved Models/' + 'abdomen_fold'+str(i)+'.pt')

        out,valid_array, omit_array = test(test_dataloader, test_model)
        print("OUT SIZEEEE",len(out))

        out = np.concatenate(out).ravel()

        np.save(args.save_path + 'Saved Models/' + f'test_predicted_label_fold{i}.npy', out)
        np.save(args.save_path + 'Saved Models/' + f'test_true_label_fold{i}.npy', test_label)

        visualize(test_label, out, i)
if iftest:
    # test
    M1_test_data_path = args.data_path + f"CV_1_M1/test_data.pt"
    M1_test_label_path = args.data_path + f"CV_1_M1/test_label.pt"
    M2_test_data_path = args.data_path + f"CV_1_M2/test_data.pt"
    M2_test_label_path = args.data_path + f"CV_1_M2/test_label.npy"

    M1_test_data = torch.load(M1_test_data_path)
    M1_test_label = torch.load(M1_test_label_path)
    M2_test_data = torch.load(M2_test_data_path)
    # M2_test_label = torch.load(M2_test_label_path)
    M2_test_label = np.load(M2_test_label_path, allow_pickle=True)
    M2_test_label = M2_test_label.item()
    M2_test_label = torch.tensor(M2_test_label["abdomen"])

    test_data = torch.cat((M1_test_data, M2_test_data), 0)
    test_label = torch.cat((M1_test_label, M2_test_label), 0)

    test_dataset = VectorCamDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for i in range(1, args.fold + 1):

        test_model = torch.load(args.save_path + 'Saved Models/' + 'abdomen_fold'+str(i)+'.pt')
        print("****Loading Fold{i} Model for Testing****", i)
        print("Cutoff Threshold: ", args.threshold)

        out, valid, omit = test(test_dataloader, test_model, threshold=args.threshold)
        out = np.concatenate(out)

        omit_count=0
        label_match=0
        label_mismatch=0
        whole_removed=[[0 for k in range(2)] for l in range(2)] # 5x5 matrix empty

        for j in range(len(omit)):
            if(omit[j]==0):
                if(out[j]==test_label[j]):
                    label_match+=1 
                    print(out[j])
                    print(whole_removed[0][0])
                    whole_removed[out[j]][out[j]]+=1

                else:
                    label_mismatch+=1
                    #print("LABEL MATCH",out[j],test_label[j])
                    #print("TYPEEE",type(test_label[j]))
                    test_label_np=test_label[j]
                    print("Test label np",test_label_np.size())
                    print("Out", len(out))
                    whole_removed[out[j]][(test_label_np)]+=1


        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()
        hm = sn.heatmap(whole_removed, annot=True, ax=ax, cmap="PuBu")
        ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
        ax.set_xlabel('Predicted labels', fontsize=15)
        ax.set_ylabel('True labels', fontsize=15)
        ax.xaxis.set_ticklabels(abdomen_categories)
        ax.yaxis.set_ticklabels(abdomen_categories)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title("Removed Confusion Matrix")
        fig_name = "Removed_CF_Fold" + str(i) +".jpg"
        plt.savefig(args.save_path + 'Results/Plots/' + fig_name)

        #print(whole_removed)
        y_axis = [label_match, label_mismatch]
        x_axis = ["match", "mismatch"]
        removed = np.array([label_match, label_mismatch])
        print('\'true labels matching with predicted label\' count: ',label_match)
        print('\'true label do not match with predicted label\' count: ',label_mismatch)


        omit_count_total = np.size(valid) - np.count_nonzero(valid)
        print("removed count", omit_count_total)
        out = out[valid]
        test_label = np.array(test_label)[valid]
        visualize(test_label, out, i)


    # majority voting
    if majority:
        test_model1 = torch.load(args.save_path + 'Saved Models/abdomen_fold1.pt')
        test_model2 = torch.load(args.save_path + 'Saved Models/abdomen_fold2.pt')
        test_model3 = torch.load(args.save_path + 'Saved Models/abdomen_fold3.pt')
        test_model4 = torch.load(args.save_path + 'Saved Models/abdomen_fold4.pt')
        test_model5 = torch.load(args.save_path + 'Saved Models/abdomen_fold5.pt')

        out1 = test(test_dataloader, test_model1)
        out1 = np.concatenate(out1)
        out2 = test(test_dataloader, test_model2)
        out2 = np.concatenate(out2)
        out3 = test(test_dataloader, test_model3)
        out3 = np.concatenate(out3)
        out4 = test(test_dataloader, test_model4)
        out4 = np.concatenate(out4)
        out5 = test(test_dataloader, test_model5)
        out5 = np.concatenate(out5)

        final_result = np.full(len(out1), 7)

        for i in range(len(final_result)):
            results = [out1[i], out2[i], out3[i], out4[i], out5[i]]
            c = Counter(results)
            value, count = c.most_common()[0]
            final_result[i] = value

        visualize(np.array(test_label), final_result, majority_voting)

    else:
        test_model = torch.load(args.save_path + 'Saved Models/abdomen_fold4.pt')
        out, valid = test(test_dataloader, test_model, threshold=0.9)
        out = np.concatenate(out)

        omit_count = np.size(valid) - np.count_nonzero(valid)

        print("removed count", omit_count)

        out = out[valid]
        test_label = np.array(test_label)[valid]

        visualize(test_label, out, 4)