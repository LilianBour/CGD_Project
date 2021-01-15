import torch
from PIL import Image
import torchvision.transforms as transforms
from  torch.utils.data import DataLoader
import cv2

def Data_Load(data_name,batch_size=128):
    if data_name=="CUB200":
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
        #LOAD DATA
        full_path="Data/CUB_200_2011/images/"
        images_names_file = open(r"Data/CUB_200_2011/images.txt")
        images_label_file = open(r"Data/CUB_200_2011/image_class_labels.txt")
        train_test_split_file= open(r"Data/CUB_200_2011/train_test_split.txt")
        label_names_file= open(r"Data/CUB_200_2011/classes.txt")
        LabelNb_LabelName=[]
        Image_Label_train=[]
        Image_Label_test=[]
        for i in label_names_file:
            LabelNb_LabelName.append(tuple((i.split()[0],i.split()[1])))
        c=0
        for (i,j,k) in zip(images_names_file,images_label_file,train_test_split_file):
            img=Image.open(full_path+i.split()[1]).convert('RGB')
            #img=cv2.imread(full_path+i.split()[1])
            #Image_Lab=[i.split()[0],tuple((img,j[-4:-1]))] (ID,(IMG,LABEL)
            if k.split()[1]=='1':
                img = transform_train(img)
                Image_Lab = [img,torch.as_tensor(int(j.split()[1]),dtype=torch.int64)]  # ((IMG,LABEL)
                Image_Label_train.append(Image_Lab)
            if k.split()[1]=='0':
                img = transform_test(img)
                Image_Lab = [img, torch.as_tensor(int(j.split()[1]),dtype=torch.int64)]  # ((IMG,LABEL)
                Image_Label_test.append(Image_Lab)
            c+=1
            #print("Loading images : ",round(c/11788*100,2),"/100%")
            #if c ==1000:#TODO Remove later just to test net
                #break
        Len_train=len(Image_Label_train)
        Len_test=len(Image_Label_test)
        Train_Loader = DataLoader(Image_Label_train, batch_size=batch_size, shuffle=False, num_workers=0)
        Test_Loader = DataLoader(Image_Label_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return Train_Loader,Test_Loader,Len_train,Len_test,LabelNb_LabelName


if __name__ == '__main__':
    t,tt=Data_Load("CUB200")
    print(len(t),len(tt))
    for i in t :
        print(i)