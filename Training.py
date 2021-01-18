import pandas as pd
import torch
from thop import profile, clever_format
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from CGD import CGD
from torch.nn import functional
from LoadDatasets import Data_Load
import argparse
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
# PART 1 --Functions--
class BatchTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    @staticmethod
    def get_anchor_positive_triplet_mask(target):
        mask = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask.fill_diagonal_(False)
        return mask

    @staticmethod
    def get_anchor_negative_triplet_mask(target):
        labels_equal = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = ~ labels_equal
        return mask

    def forward(self, x, target):
        pairwise_dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        mask_anchor_positive = self.get_anchor_positive_triplet_mask(target)
        anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(target)
        # make positive and anchor to be exclusive through maximizing the dist
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]

        loss = (functional.relu(hardest_positive_dist - hardest_negative_dist + self.margin))
        return loss.mean()


# PART 2 --Train--
#Set param various
epochs = 2
batch = 32
temperature=0.5
margin=0.1
evaluation={'loss':[],'acc':[]}

#Load data
Data_Name="IRMA_XRAY"
Train_loader, Validation_Loader,Test_loader, Len_train, Len_val,Len_test, LabelNb_LabelName, Image_Label_test,ImageName_Idx_Test =Data_Load(Data_Name,batch)
#Define model (CGD here)
Dim=1536
Global_Descriptors = ['S','G','M']
nb_classes=len(LabelNb_LabelName)+1 #TODO +1???
model=CGD(Global_Descriptors,Dim,nb_classes).to(device)
optimizer=Adam(model.parameters(),lr=1e-4)#1e-4 was the start changed because loss kept increasing but with 1e-8 accuracy increase slowly
step_decay = MultiStepLR(optimizer, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1) #Change the lr at 50% a,d 75% (not indicated in the article)
Triplet_Loss = BatchTripletLoss(margin=margin) #TODO

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
        for image,label in tqdm(Train_loader):
            image = image.to(device)
            label = label.to(device)
            GDs,Cl=model(image)
            Rk_loss = Triplet_Loss(GDs, label)  # TODO
            CrossLoss = nn.CrossEntropyLoss()#TODO add the use of temperature ??
            Cl_loss = CrossLoss(Cl,label)
            #triplet_loss = nn.TripletMarginLoss(margin=margin,p=2) or Soft Margin Loss
            #nn.SoftMarginLoss()
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
                Rk_loss = Triplet_Loss(GDs, label)
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


    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Plot Losses
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

#TODO 1.check todos + review
#TODO 2.add other datasets

#TODO add recall an precision ?
#TODO try to set batch to 128