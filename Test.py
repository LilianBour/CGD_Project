import os
import shutil
import torch
from PIL import Image, ImageDraw
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

if __name__ == '__main__':
    Data_Name = "CUB200"
    full_path = "Data/CUB_200_2011/images/"
    # PART 1 --Test--
    # Load data
    Triplet_Loss = BatchTripletLoss(margin=0.1)  # TODO
    batch = 32
    Train_loader, Validation_Loader,Test_loader, Len_train, Len_val,Len_test, LabelNb_LabelName, Image_Label_test,ImageName_Idx_Test = Data_Load(Data_Name, batch)
    Dim = 1536
    Global_Descriptors = ['S', 'G', 'M']
    nb_classes = len(LabelNb_LabelName) + 1

    #Load model
    model = CGD(Global_Descriptors, Dim, nb_classes).to(device)
    path = "C:\\Users\\lilia\\github\\CGD_Project\\Models\\"
    name = "model_21_CUB200.pt"
    model.load_state_dict(torch.load(path + name))

    #Set data for image retrieval
    data_test_images =[]
    data_test_labels = []
    for (i,j) in zip(Image_Label_test,ImageName_Idx_Test):
        data_test_images.append(full_path+j[0])
        data_test_labels.append(i[1])
    data_test_features = []

    if __name__ == "__main__":
        best_loss = float('inf')
        train_loss = []
        accuracy_list = []
        with torch.no_grad():
            model.eval()
            train_loss.append(0)
            accuracy_list.append(0)
            TP = 0
            T = 0
            for image, label in tqdm(Test_loader):
                image = image.to(device)
                label = label.to(device)
                GDs, Cl = model(image)
                data_test_features.append(GDs)
                Rk_loss = Triplet_Loss(GDs, label)
                CrossLoss = nn.CrossEntropyLoss()
                Cl_loss = CrossLoss(Cl, label)
                T_loss = Rk_loss + Cl_loss
                pred = torch.argmax(Cl, dim=-1)
                TP += torch.sum(pred == label).item()
                T += image.size(0)
                train_loss[0] = T_loss.item() * image.size(0)
            train_loss[0] = train_loss[0] / Len_train
            accuracy_list[0] = TP / T
            data_test_features=torch.cat(data_test_features,dim=0)
            print("Epoch ", 0, "; train loss = ", train_loss[0], "; Accuracy = ", accuracy_list[0])


    # PART 2 --Image Retrieval--
    img_name = "017.Cardinal/Cardinal_0047_17673.jpg"
    #img_name = "101.White_Pelican/White_Pelican_0025_97604.jpg"
    query_img_name= full_path+img_name
    #data_base_name= "C:\\Users\\lilia\\github\\CGD_Project\\Models\\data_0_" + Data_Name + ".pth"
    retrieval_num = 10
    #data_base = torch.load(data_base_name)

    #find query index
    query_index=0
    for i in ImageName_Idx_Test:
        if i[0]==img_name:
            print(img_name)
            query_index=i[1]

    query_image = Image.open(query_img_name).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
    query_label = data_test_labels[query_index] #TODO torch.tensor useless because data_test_labels already tensors
    query_feature = data_test_features[query_index]

    dist_matrix = torch.cdist(query_feature.unsqueeze(0).unsqueeze(0), data_test_features.unsqueeze(0)).squeeze()
    dist_matrix[query_index] = float('inf')
    idx = dist_matrix.topk(k=retrieval_num, dim=-1, largest=False)[1]


    retrieval_path = 'Image_retrieval/{}'.format(query_img_name.split('/')[-1].split('.')[0])
    if os.path.exists(retrieval_path):
        shutil.rmtree(retrieval_path)
    os.mkdir(retrieval_path)
    query_image.save('{}/query_img.jpg'.format(retrieval_path))
    for num, index in enumerate(idx):
        retrieval_image = Image.open(data_test_images[index.item()]).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(retrieval_image)
        retrieval_label = data_test_labels[index.item()]
        retrieval_status = (retrieval_label == query_label).item()
        retrieval_dist = dist_matrix[index.item()].item()
        if retrieval_status:
            draw.rectangle((0, 0, 223, 223), outline='green', width=8)
        else:
            draw.rectangle((0, 0, 223, 223), outline='red', width=8)
        retrieval_image.save('{}/retrieval_img_{}_{}.jpg'.format(retrieval_path, num + 1, '%.4f' % retrieval_dist))