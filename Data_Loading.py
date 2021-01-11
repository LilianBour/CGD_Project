import tensorflow as tf
from sklearn.model_selection import train_test_split

class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website
class_nb=len(class_types)

#-- DATA LOADING --
(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.cifar10.load_data()

#Normalize images to pixel val
train_img, test_img = train_img / 255.0 , test_img / 255.0
#Format and shape
print ("Types of train_im/train_lab  ", type(train_img), type(train_labels))
print ("Shape of images/labels : ", train_img.shape, train_labels.shape)
print ("Shape of images/labels (test) ", test_img.shape, test_labels.shape)

#Unique elem distribution maybe delete later
'''
(unique, counts) = np.unique(train_lab, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print (frequencies)
print (len(unique))
'''

#Print some images with label maybe delete later
'''
plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_im[i], cmap='gray')
    plt.xlabel(class_types[train_lab[i][0]], fontsize=13)
plt.tight_layout()
plt.show()
'''

#One hot encoding (labels)
train_lab_categorical = tf.keras.utils.to_categorical(train_labels, num_classes=class_nb, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(test_labels, num_classes=class_nb, dtype='uint8')

#Data Split
train_img, valid_im, train_labels, valid_lab = train_test_split(train_img, train_lab_categorical, test_size=0.20, stratify=train_lab_categorical, random_state=40, shuffle = True)
train_img, train_labels= train_img[0:1000], train_labels[0:1000] #Take few training data to test faster
print ("train data shape after the split: ", train_img.shape)
print ('new validation data shape: ', valid_im.shape)
print ("validation labels shape: ", valid_lab.shape)


#-- DATA AUGMENTATION --
batch_size = 64
number_epochs = 4
train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
train_set_conv = train_DataGen.flow(train_img, train_labels, batch_size=batch_size)  # train_lab is categorical
valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size)  # so as valid_lab