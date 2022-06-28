from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn

def visualize(true_label, predicted_label, output_path, species_all, i = 0):
    # print classification report
    #print(true_label)
    print(len(true_label))
    print(np.size(predicted_label))
    print(classification_report(true_label, predicted_label))
    #species_all = ["An. funestus","An. gambiae", "An. 'other' ", "Cx.", "An. stephensi", "other"]
    species_all = ["An. funestus","An. gambiae", "An. 'other' ", "Cx.", "other"]
    # samples count confusion matrix
    conf_mat = confusion_matrix(true_label, predicted_label)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    hm = sn.heatmap(conf_mat, annot=True, ax=ax, cmap="PuBu",fmt = "d")
    ax.set_yticklabels(hm.get_yticklabels(), rotation=90)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    ax.xaxis.set_ticklabels(species_all)
    ax.yaxis.set_ticklabels(species_all)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    #print(conf_mat)
    if not os.path.exists(output_path+'Test/'):
        os.makedirs(output_path+'Test/')
    plt.savefig(output_path+'Test/CountCF_'+str(i)+'.jpg')

    #accuracy confusion matrix - horizontal
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
    if not os.path.exists(output_path+'Test/'):
        os.makedirs(output_path+'Test/')
    plt.savefig(output_path+'Test/CF_h_'+str(i)+'.jpg')

    #accuracy confusion matrix - vertical
    conf_mat = conf_mat / np.expand_dims(conf_mat.astype(np.float64).sum(axis=0), 1)
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
    if not os.path.exists(output_path+'Test/'):
        os.makedirs(output_path+'Test/')
    plt.savefig(output_path+'Test/CF_v_'+str(i)+'.jpg')

def crop(im):
    """
    input the image, return the cropped part of the image
    : im: image to crop
    : returns cropped image
    """
    # Size of the image in pixels (size of original image)
    width, height = im.size

    if width > height:
        diff = width - height
        top = 0
        bottom = height
        left = diff//2
        right = width - (diff//2)
        crop_img = im.crop((left, top, right, bottom))

    elif height > width:
        diff = height - width
        top = diff//2
        bottom = height - (diff//2)
        left = 0
        right = width
        crop_img = im.crop((left, top, right, bottom))
    else:
        crop_img = im
    
    return crop_img

def pad(im, color):
    """
    Pads the image to make it a square"
    : im: image to pad
    : color: color to pad the image
    : returns padded image
    """
    width, height = im.size

    # if widht and height are same, padding is not required.
    if width == height:
        pad_im = im
    
    elif width > height:
        pad_img = Image.new(im.mode, (width, width), color)
        pad_img.paste(im, (0, (width - height) // 2))
    
    else:
        pad_img = Image.new(im.mode, (height, height), color)
        pad_img.paste(im, ((height - width)//2, 0))
    
    return pad_img

def getClassWeights(train_dataset):
    target_list = torch.tensor(train_dataset.labels)
    class_count = np.array([len(np.where(train_dataset.labels == t)[0]) for t in np.unique(train_dataset.labels)])
    print("Class count :", class_count)
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    print("Class weights",class_weights)
    print("\n")
    class_weights_all = class_weights[target_list]
    return class_weights_all

