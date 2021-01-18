"""
train_file = open(r"Data/In_Shop_Clothes/In_Shop_train.txt")
test_file = open(r"Data/In_Shop_Clothes/In_Shop_test.txt")
test_lab=[]
train_lab=[]
for i in train_file:
    label= int(i.split()[1])
    if not label in train_lab:
        train_lab.append(label)


for i in test_file:
    label= int(i.split()[1])
    if not label in test_lab:
        test_lab.append(label)

print(len(test_lab)," ",len(train_lab))
print(test_lab,"\n",train_lab)
"""
"""
#Get number of classes in IRMA_XRAY
from _csv import reader
test_lab=[]
with open('Data/IRMA_XRAY/ImageCLEFmed2009_train.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    c=0
    next(csv_reader)  # Skip the header
    for row in csv_reader:
        label = row[0].split(";")[2]
        if label == '\\N':
            c = c  # CONTINUE not working ?
        else:
            label = int(label)
            if not label in test_lab:
                test_lab.append(label)

train_lab=[]
with open('Data/IRMA_XRAY/ImageCLEFmed2009_test.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    c=0
    next(csv_reader)  # Skip the header
    for row in csv_reader:
        label = row[0].split(";")[6]
        if label == '\\N':
            c = c  # CONTINUE not working ?
        else:
            label = int(label)
            if not label in train_lab:
                train_lab.append(label)
print(len(test_lab)," ",len(train_lab))
print(test_lab,"\n",train_lab)
x=[]
for i in range(57):
    x.append(i)
print(len(x))
"""
"""
#InShop create txt [img name, label, train/test] to load data without needing to loop both files
write_train = open(r"Data/In_Shop_Clothes/In_Shop_train.txt","w+")
write_test = open(r"Data/In_Shop_Clothes/In_Shop_test.txt","w+")
to_write_train=[]
to_write_test=[]
c=0
with  open("Data/In_Shop_Clothes/list_bbox_inshop.txt","r+") as label:
    for i in label:
        print((c/52712)*100)
        c=c+1
        img_name=i.split()[0]
        img_label=i.split()[1]
        img_eval= "-"
        with open("Data/In_Shop_Clothes/list_eval_partition.txt",'r+') as evalu:
            for ev in evalu:
                if i.split()[0]==ev.split()[0]:
                    if ev.split()[2]=="train":
                        img_eval="train"
                        to_write_train.append([img_name + " " + img_label + " " + img_eval + "\n"])
                    elif ev.split()[2]=="gallery" or ev.split()[2]=="query" :
                        img_eval="test"
                        to_write_test.append([img_name + " " + img_label + " " + img_eval + "\n"])



for L in to_write_train:
    write_train.writelines(L)
for L in to_write_test:
    write_test.writelines(L)
"""