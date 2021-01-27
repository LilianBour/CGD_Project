import os
import shutil
from PIL import Image, ImageDraw
import torch
from torch import nn
from tqdm import tqdm
from CGD import CGD
from torch.nn import functional
from LoadDatasets import Data_Load
from TripleLoss import batch_hard_triplet_loss
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()


if __name__ == '__main__':
    Data_Name = "CUB_200_2011"
    Model_NB="22"
    #--QUERY IMAGE--
        #CUB_200_2011
    img_name = ["images/017.Cardinal/Cardinal_0047_17673.jpg","images/078.Gray_Kingbird/Gray_Kingbird_0004_70293.jpg","images/080.Green_Kingfisher/Green_Kingfisher_0028_70981.jpg","images/161.Blue_winged_Warbler/Blue_Winged_Warbler_0060_161888.jpg","images/101.White_Pelican/White_Pelican_0025_97604.jpg"]

        #New_CUB_200_2011
    #img_name=["images/101.White_Pelican/White_Pelican_0077_97025.jpg","images/109.American_Redstart/American_Redstart_0090_102940.jpg","images/117.Clay_colored_Sparrow/Clay_Colored_Sparrow_0072_110851.jpg","images/171.Myrtle_Warbler/Myrtle_Warbler_0023_166764.jpg","images/173.Orange_crowned_Warbler/Orange_Crowned_Warbler_0097_168004.jpg","images/185.Bohemian_Waxwing/Bohemian_Waxwing_0092_796666.jpg"]

        #Stanford_Online_Products
    #img_name=["bicycle_final/251952414262_5.JPG","chair_final/351235374628_6.JPG","chair_final/321078646383_0.JPG","mug_final/291392767147_7.JPG","mug_final/252048327345_3.JPG","lamp_final/231645131622_0.JPG","lamp_final/252046443465_2.JPG"]
        #CARS196
    #img_name=["cars_test/00021.jpg",img_name="cars_test/00913.jpg","cars_test/01306.jpg","cars_test/02389.jpg"]

        #In Shop
    #img_name=["img/WOMEN/Tees_Tanks/id_00000099/05_7_additional.jpg","img/WOMEN/Pants/id_00000105/02_2_side.jpg","img/WOMEN/Shorts/id_00000144/03_3_back.jpg","img/MEN/Sweatshirts_Hoodies/id_00000146/02_1_front.jpg "]

        #IRMA XRAY
    #img_name=["test_img/11501.png","test_img/12570.png","test_img/12688.png"]

    full_path = "Data/"+Data_Name+"/"
    if Data_Name=="New_CUB_200_2011":full_path="Data/CUB_200_2011/"
    # PART 1 --Test--
    # Load data
    batch = 32
    margin = 0.1
    Train_loader, Validation_Loader,Test_loader, Len_train, Len_val,Len_test, LabelNb_LabelName, Image_Label_test,ImageName_Idx_Test = Data_Load(Data_Name, batch,T="test")
    Dim = 1536
    Global_Descriptors = ['S', 'G', 'M']
    nb_classes = len(LabelNb_LabelName) +1

    #Load model
    model = CGD(Global_Descriptors, Dim, nb_classes).to(device)
    path = "C:\\Users\\lilia\\github\\CGD_Project\\Models\\"+Data_Name+"\\"
    name = "model_"+Model_NB+"_"+Data_Name+".pt"
    model.load_state_dict(torch.load(path + name))

    #Set data for image retrieval
    data_test_images =[]
    data_test_labels = []
    for (i,j) in zip(Image_Label_test,ImageName_Idx_Test):
        #data_test_images.append(full_path+j[0])
        data_test_images.append(j[0])
        data_test_labels.append(i[1])
    data_test_features = []

    best_loss = float('inf')
    test_loss = []
    accuracy_list = []
    with torch.no_grad():
        model.eval()
        test_loss.append(0)
        accuracy_list.append(0)
        TP = 0
        T = 0
        for image, label in tqdm(Test_loader):
            image = image.to(device)
            label = label.to(device)
            GDs, Cl = model(image)
            data_test_features.append(GDs)
            Rk_loss = batch_hard_triplet_loss(label,GDs,margin=margin)
            CrossLoss = nn.CrossEntropyLoss()
            Cl_loss = CrossLoss(Cl, label)
            T_loss = Rk_loss + Cl_loss
            pred = torch.argmax(Cl, dim=-1)
            TP += torch.sum(pred == label).item()
            T += image.size(0)
            test_loss[0] = T_loss.item() * image.size(0)
        test_loss[0] = test_loss[0] / Len_test
        accuracy_list[0] = TP / T
        data_test_features=torch.cat(data_test_features,dim=0)
        print("Epoch ", 0, "; test loss = ", test_loss[0], "; Accuracy = ", accuracy_list[0])


    # PART 2 --Image Retrieval--
    for j in img_name:
        query_img_name= full_path+j
        retrieval_num = 10
        #find query index
        query_index=0
        for i in ImageName_Idx_Test:
            #print(i[0],' ,',full_path+img_name)
            if i[0]==full_path+j:
                print(i)
                query_index=i[1]

        query_image = Image.open(query_img_name).convert('RGB').resize((224, 224), resample=Image.BILINEAR)
        query_label = torch.tensor(data_test_labels[query_index])
        query_feature = data_test_features[query_index]


        dist_matrix = torch.cdist(query_feature.unsqueeze(0).unsqueeze(0), data_test_features.unsqueeze(0)).squeeze()
        dist_matrix[query_index] = float('inf')
        idx = dist_matrix.topk(k=retrieval_num, dim=-1, largest=False)[1]
        #Find and save best images results
        retrieval_path = 'Image_retrieval/'+Data_Name+'/{}'.format(query_img_name.split('/')[-1].split('.')[0])

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