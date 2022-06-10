import os
import torch
import numpy as np
import torch.nn.functional as F

def train(model,  criterion, optimizer, train_dataloader, output_path, EPOCHS = 10, i=0):
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    max_val = 0

    for epoch in range(EPOCHS):

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
        print(f"Epoch: {epoch+1},Train Loss: {train_ave_loss/len(train_dataloader)} | Train Acc: {100.*correct/total} ({correct}/{total})")

        model.eval()

        if not os.path.exists(output_path+'Train/'):
            os.makedirs(output_path+'Train/')
        torch.save(model,output_path+'Train/Model_'+str(i)+'.pt')
    
    return train_loss, valid_loss, train_acc, valid_acc

def test(model, test_dataloader, threshold = 0.8):
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