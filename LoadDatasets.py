import random
import torch
from csv import reader
from PIL import Image
import torchvision.transforms as transforms
from  torch.utils.data import DataLoader
import scipy.io

#Load data for CUB_200_2011, New_CUB_200_2011, CARS196, In_Shop_Clothes or Stanford_Online_Products
def Data_Load(data_name,batch_size=128,T="train"):
    #Transform for train and test, Validation has train_transform applied
    transform_train = transforms.Compose(
            [transforms.Resize((254,254)),
            transforms.RandomCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    transform_test = transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    LabelNb_LabelName = []
    Image_Label_training = []
    Image_Label_test = []
    ImageName_Idx_Test = []
    Image_Label_train = []
    Image_Label_val = []

    if data_name=="CUB_200_2011": #OVERFITTING not enough data for the training part maybe
        full_path="Data/CUB_200_2011/images/"
        #Load data
        images_names_file = open(r"Data/CUB_200_2011/images.txt")
        images_label_file = open(r"Data/CUB_200_2011/image_class_labels.txt")
        train_test_split_file= open(r"Data/CUB_200_2011/train_test_split.txt")
        label_names_file= open(r"Data/CUB_200_2011/classes.txt")

        #Define a list that will allow to have the number of class, here we have a list with the label associated to each class
        for i in label_names_file:
            LabelNb_LabelName.append(tuple((i.split()[0],i.split()[1])))
        c=0
        idx=0
        #Create Train and Test [[PIL img, label as int for Train and Tensor for Test],...]
        for (i,j,k) in zip(images_names_file,images_label_file,train_test_split_file):
            img=Image.open(full_path+i.split()[1]).convert('RGB')
            #img=cv2.imread(full_path+i.split()[1]) #Reading image with OpenCv, not working because image needed as PIL format
            if k.split()[1]=='1' and T=="train":
                img = transform_train(img)
                Image_Lab = [img,int(j.split()[1])]  # ((IMG,LABEL) // torch.as_tensor(int(j.split()[1]),dtype=torch.int64)
                Image_Label_training.append(Image_Lab)
            if k.split()[1]=='0' and T=="test":
                ImageName_Idx_Test.append([full_path+i.split()[1],idx])
                idx=idx+1
                img = transform_test(img)
                Image_Lab = [img, int(j.split()[1])]  #label as tensor needed for the test part, specifically the part where the ranked image are found and saved
                Image_Label_test.append(Image_Lab)
            c+=1
            #print("Loading images : ",round(c/11788*100,2),"/100%") #Show data loading progression


    if data_name == "New_CUB_200_2011": #Issue with labels, weird behaviour (see Image retrieval folder)
        full_path = "Data/CUB_200_2011/"
        train_file = open(r"Data/New_CUB_200_2011/train.txt")
        test_file = open(r"Data/New_CUB_200_2011/test.txt")

        #Only 100 classes because 100first classes -> training and 100-200 classes - testing
        for i in range(100):
            LabelNb_LabelName.append(i)
        if T=="train":
            c=0
            for i in train_file:
                img = Image.open(full_path + i.split()[2]).convert('RGB')
                img = transform_train(img)
                Image_Lab = [img, int(i.split()[1])]
                Image_Label_training.append(Image_Lab)
                c = c + 1
                if c == 1000:  # TODO Remove later just to test net
                    break
        if T=="test":
            idx=0
            for i in test_file:
                img = Image.open(full_path + i.split()[2]).convert('RGB')
                #ImageName_Idx_Test.append([full_path+i.split()[2], int(i.split()[0])]) TEST MANUAL IDX
                ImageName_Idx_Test.append([full_path+i.split()[2], idx])
                idx=idx+1
                img = transform_test(img)
                Image_Lab = [img, int(i.split()[1])]
                Image_Label_test.append(Image_Lab)


    if data_name == "Stanford_Online_Products":
        full_path = "Data/Stanford_Online_Products/"
        train_file = open(r"Data/Stanford_Online_Products/Ebay_train.txt")
        test_file = open(r"Data/Stanford_Online_Products/Ebay_test.txt")

        LabelNb_LabelName = ["bicycle", "cabinet", "chair", "coffee maker", "fan", "kettle", "lamp", "mug", "sofa", "stapler", "table", "toaster"]
        if T=="train":
            c=0
            for i in train_file:
                img = Image.open(full_path + i.split()[3]).convert('RGB')
                img = transform_train(img)
                Image_Lab = [img, int(i.split()[2])]
                Image_Label_training.append(Image_Lab)
                c=c+1
                if c == 1000:  # TODO Remove later just to test net
                    break
        if T=="test":
            c=0
            for i in test_file:
                img = Image.open(full_path + i.split()[3]).convert('RGB')
                ImageName_Idx_Test.append([full_path+i.split()[3], i.split()[0]])
                img = transform_test(img)
                Image_Lab = [img, int(i.split()[2])]
                Image_Label_test.append(Image_Lab)
                c = c + 1
                if c == 1000:  # TODO Remove later just to test net
                    break

    if data_name=="CARS196":
        #Maybe probel with number of classes
        for i in range(196):
            LabelNb_LabelName.append(i)
        if T=="train":
            full_path_train="Data/CARS196/cars_train/"
            train = scipy.io.loadmat('Data/CARS196/cars_train_annos.mat')
            anno_train = train['annotations']
            for i in anno_train:
                c=0
                for j in i:
                    img_path = full_path_train+j[5][0]
                    label = int(j[4][0][0])
                    img = Image.open(img_path).convert('RGB')
                    img = transform_train(img)
                    Image_Lab = [img, label]
                    Image_Label_training.append(Image_Lab)
                    c = c + 1
                    if c == 1000:  # TODO Remove later just to test net
                        break
        if T=="test":
            full_path_test = "Data/CARS196/cars_test/"
            test = scipy.io.loadmat('Data/CARS196/cars_test_annos.mat')
            anno_test = test['annotations']
            idx=0
            for i in anno_test:
                c=0
                for j in i:
                    img_path = full_path_test + j[5][0]
                    label = int(j[4][0][0])
                    img = Image.open(img_path).convert('RGB')
                    ImageName_Idx_Test.append([img_path, idx])
                    idx=idx+1
                    img = transform_test(img)
                    Image_Lab = [img, label]
                    Image_Label_test.append(Image_Lab)
                    c = c + 1
                    if c == 1000:  # TODO Remove later just to test net
                        break

    if data_name=="In_Shop_Clothes":
        full_path = "Data/In_Shop_Clothes/"
        train_file = open(r"Data/In_Shop_Clothes/In_Shop_train.txt")
        test_file = open(r"Data/In_Shop_Clothes/In_Shop_test.txt")

        # Only 100 classes because 100first classes -> training and 100-200 classes - testing
        for i in range(3):
            LabelNb_LabelName.append(i)
        if T == "train":
            c=0
            for i in train_file:
                img = Image.open(full_path + i.split()[0]).convert('RGB')
                img = transform_train(img)
                Image_Lab = [img, int(i.split()[1])]
                Image_Label_training.append(Image_Lab)
                c = c + 1
                if c == 1000:  # TODO Remove later just to test net
                    break
        if T == "test":
            idx = 0
            c=0
            for i in test_file:
                img = Image.open(full_path + i.split()[0]).convert('RGB')
                ImageName_Idx_Test.append([full_path + i.split()[0], idx])
                idx = idx + 1
                img = transform_test(img)
                Image_Lab = [img, int(i.split()[1])]
                Image_Label_test.append(Image_Lab)
                c = c + 1
                if c == 1000:  # TODO Remove later just to test net
                    break

    if data_name == "IRMA_XRAY":
        if T=="train":
            for i in range(57):
                LabelNb_LabelName.append(i)
            full_path_train="Data/IRMA_XRAY/train_img/"
            with open('Data/IRMA_XRAY/ImageCLEFmed2009_train.csv', 'r') as read_obj:
                csv_reader = reader(read_obj)
                next(csv_reader) #Skip the header
                c=0
                for row in csv_reader:
                    img_path= str(full_path_train+row[0].split(";")[0]+".png")
                    label= row[0].split(";")[2]
                    img = Image.open(img_path).convert('RGB')
                    img = transform_train(img)
                    if label == '\\N':
                        c=c #CONTINUE not working ?
                    else :
                        label = int(label)
                        Image_Lab = [img, label]
                        Image_Label_training.append(Image_Lab)
                    c = c + 1
                    if c == 1000:  # TODO Remove later just to test net??
                        break
        if T == "test":
            for i in range(57): #size should be 55 but it's not working when testing
                LabelNb_LabelName.append(i)
            full_path_test = "Data/IRMA_XRAY/test_img/"
            idx=0
            with open('Data/IRMA_XRAY/ImageCLEFmed2009_test.csv', 'r') as read_obj:
                csv_reader = reader(read_obj)
                next(csv_reader) #Skip the header
                c=0
                for row in csv_reader:
                    img_path = full_path_test + row[0].split(";")[0]+".png"
                    label = row[0].split(";")[6]
                    img = Image.open(img_path).convert('RGB')
                    ImageName_Idx_Test.append([img_path, idx])
                    idx = idx + 1
                    img = transform_test(img)
                    if label == '\\N':
                        c=c #CONTINUE not working ?
                    else :
                        label = int(label)
                        Image_Lab = [img, label]
                        Image_Label_test.append(Image_Lab)
                    c = c + 1



    #Creating validation set from test set. Test = 80%, Validation = 20%
    if T == "train":
        random.shuffle(Image_Label_training)
        Image_Label_train = Image_Label_training[:int((0.8 * len(Image_Label_training)))]
        Image_Label_val = Image_Label_training[int((0.8 * len(Image_Label_training))):]
        Train_Loader = DataLoader(Image_Label_train, batch_size=batch_size, shuffle=True, num_workers=0)
        Validation_Loader = DataLoader(Image_Label_val, batch_size=batch_size, shuffle=True, num_workers=0)
        Test_Loader = []
    if T == "test":
        Train_Loader = []
        Validation_Loader = []
        Test_Loader = DataLoader(Image_Label_test, batch_size=batch_size, shuffle=False, num_workers=0)
    # Getting len of Datasets
    Len_train = len(Image_Label_train)
    Len_val = len(Image_Label_val)
    Len_test = len(Image_Label_test)
    return Train_Loader,Validation_Loader,Test_Loader,Len_train,Len_val,Len_test,LabelNb_LabelName,Image_Label_test,ImageName_Idx_Test


if __name__ == '__main__':
    #Test code here
    print("Test")