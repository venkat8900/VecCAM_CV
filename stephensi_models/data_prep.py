import os 
import glob
import random
import pandas as pd

def find_label(row,label_set):
  label = row.split('/')[5]
  # print(label)
  if label not in label_set:
      label = 'other'
  indexLabel = label_set.index(label)
  return indexLabel 


def specimen_segregator(path, store_path, endWith, label_set, train_test_split):

    files = []
    count = 0
    for root, dirs, mosfile in os.walk(path):
        for file in mosfile:
            if file.endswith(endWith):
                files.append(root+'/'+file)
                count+=1
    print("total number of images :", count)

    ## Create a set with all unique IDs of the mosquito
    ID = set();
    for file in files:
        name = file.split('/')[-1].split('_')[0]
        ID.add(name)

    ## Create a dictionary of specimen to store all relevant images pertaining to a specimen
    species_dict = {}
    for id in ID:
        species_dict[id] = []

    ## Iterate through all filenames and arrange them according to specimen number
    for fname in files:
        specimen = fname.split('/')[-1].split('_')[0]
        species_dict[specimen].append(fname);

    ## Lists to store train and test specimen
    test_data = []
    train_data = []
    count = 1
    total = len(ID)

    ## Iterate through every ID and based on random split value, keep it in train set or test_set
    for id in ID:
        if count%100==0:
            print(f"Specimen Number:{count}/{total}")
        count+=1
        flag = random.random()
        if flag < train_test_split:
            train_data += species_dict[id]
        else:
            test_data += species_dict[id]


    ## Convert lists to pandas dataframes
    train_dict = {'filename':train_data}
    train_df = pd.DataFrame(train_dict)

    test_dict = {'filename':test_data}
    test_df = pd.DataFrame(test_dict)

    ## Assign label to each specimen
    train_df['label']=train_df['filename'].apply(find_label,args=(label_set,))
    test_df['label']=test_df['filename'].apply(find_label,args=(label_set,))

    ## Save CSV Files
    
    train_df.to_csv(store_path+'train.csv')
    test_df.to_csv(store_path+'test.csv')
    print("CSV files saved to :", store_path)
    print('\n')

    return train_df, test_df