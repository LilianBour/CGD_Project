import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from CGD import CGD
from LoadDatasets import Data_Load
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from TripleLoss import batch_hard_triplet_loss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()

# PART 1 --Train--
#Set param various
epochs = 30
batch = 32 #128
temperature=0.5
margin=1

#Load data
Data_Name="CUB_200_2011"
Train_loader, Validation_Loader,Test_loader, Len_train, Len_val,Len_test, LabelNb_LabelName, Image_Label_test,ImageName_Idx_Test =Data_Load(Data_Name,batch)
#Define model (CGD here)
Dim=1536
Global_Descriptors = ['S','G','M']
nb_classes=len(LabelNb_LabelName)+1
model=CGD(Global_Descriptors,Dim,nb_classes).to(device)
optimizer=Adam(model.parameters(),lr=1e-4)#1e-4 was the start changed because loss kept increasing but with 1e-8 accuracy increase slowly
step_decay = MultiStepLR(optimizer, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1) #Change the lr at 50% a,d 75% (not indicated in the article) LR0
#step_decay = MultiStepLR(optimizer, milestones=[int(0.9 * epochs), int(0.95 * epochs)], gamma=0.1) #Change the lr at 90% a,d 95% (not indicated in the article) LR1
#step_decay = MultiStepLR(optimizer, milestones=[int(0.7 * epochs), int(0.90 * epochs)], gamma=0.1) #Change the lr at 90% a,d 95% (not indicated in the article) LR2


#Load Data
#Train
if __name__=="__main__":
    best_loss=float('inf')
    train_loss=[]
    accuracy_list=[]
    val_loss = []
    val_accuracy_list = []
    for epoch in range(epochs+1):
        print("Starting epoch ",epoch)
        #Train
        model.train()
        train_loss.append(0)
        accuracy_list.append(0)
        TP=0
        T=0
        List_For_RankingLoss = []
        for image,label in tqdm(Train_loader):
            image = image.to(device)
            label_nocuda = label.detach().clone()
            label = label.to(device)
            GDs,Cl=model(image)
            Rk_loss = batch_hard_triplet_loss(label,GDs,margin=margin)
            #Triplet Sampling for Triplet Loss RANDOM
            """
            TripletLoss = nn.TripletMarginLoss(margin=margin)
            list_TL = []
            for i in range(len(label)):
                list_TL.append([GDs[i],label[i]])
            random.shuffle(list_TL)

            anchor = GDs[0]
            positive = GDs
            negative = GDs
            for i in range(len(list_TL)):
                rand = random.randrange(len(list_TL))
                if label[rand]==label[i] and torch.allclose(GDs[rand], GDs[i], atol=0)==False:
                    positive = GDs[i]
                elif label[rand]!=label[i]:
                    negative=GDs[i]
            Rk_loss = TripletLoss(negative, positive, anchor)
            """

            CrossLoss = nn.CrossEntropyLoss()#TODO add the use of temperature
            Cl_loss = CrossLoss(Cl,label)
            T_loss=Rk_loss+Cl_loss
            optimizer.zero_grad()
            T_loss.backward()
            optimizer.step()

            pred = torch.argmax(Cl, dim=-1)
            TP += torch.sum(pred == label).item()
            T += image.size(0)
            train_loss[epoch]=T_loss.item()*image.size(0)
        train_loss[epoch]= train_loss[epoch]/Len_train
        accuracy_list[epoch]=TP/T
        step_decay.step()
        #Validation
        with torch.no_grad():
            model.eval()
            val_loss.append(0)
            val_accuracy_list.append(0)
            TP = 0
            T = 0
            for image, label in tqdm(Validation_Loader):
                image = image.to(device)
                label = label.to(device)
                GDs, Cl = model(image)
                Rk_loss = batch_hard_triplet_loss(label,GDs,margin=margin)
                CrossLoss = nn.CrossEntropyLoss()
                Cl_loss = CrossLoss(Cl, label)
                T_loss = Rk_loss + Cl_loss
                pred = torch.argmax(Cl, dim=-1)
                TP += torch.sum(pred == label).item()
                T += image.size(0)
                val_loss[epoch] = T_loss.item() * image.size(0)
            val_loss[epoch] = val_loss[epoch] / Len_val
            val_accuracy_list[epoch] = TP / T

        print("Epoch ", epoch, "; Train loss = ", train_loss[epoch], "; Train Accuracy = ", accuracy_list[epoch],"; Val loss = ", val_loss[epoch], "; Val Accuracy = ", val_accuracy_list[epoch], )
        if val_loss[epoch] < best_loss :
            print("Model Saved because : ",val_loss[epoch],"<",best_loss)
            best_loss=val_loss[epoch]
            torch.save(model.state_dict(),"C:\\Users\\lilia\\github\\CGD_Project\\Models\\"+Data_Name+"\\model_"+str(epoch)+"_"+Data_Name+".pt")

    # PART 2 --Plot Losses--
    plt.subplot(1, 2, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss curve")
    TRAIN_LOSS = mpatches.Patch(color='Blue', label='Training Loss')
    VALIDATION_LOSS = mpatches.Patch(color='Orange', label='Validation Loss')
    plt.legend(handles=[TRAIN_LOSS,VALIDATION_LOSS])
    plt.plot(range(len(train_loss)), train_loss,label="TRAIN_LOSS")
    plt.plot(range(len(val_loss)), val_loss,label='VALIDATION_LOSS')
    # Plot Accurarcy
    plt.subplot(1, 2, 2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("accuracy curve")
    TRAIN_ACC = mpatches.Patch(color='Blue', label='Training Accuracy')
    VALIDATION_ACC = mpatches.Patch(color='Orange', label='Validation Accuracy')
    plt.legend(handles=[TRAIN_ACC,VALIDATION_ACC])
    plt.plot(range(len(accuracy_list)),accuracy_list,label="TRAIN_ACC")
    plt.plot(range(len(val_accuracy_list)), val_accuracy_list,label='VALIDATION_ACC')
    plt.tight_layout()
    plt.show()


#TODO try to set batch to 128