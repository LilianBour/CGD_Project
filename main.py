import ResNet50
import Data_Loading
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam


#-- Functions --
def conf_matrix(predictions):
    cm=confusion_matrix(test_lab, np.argmax(np.round(predictions), axis=1))
    print("Classification Report:\n")
    cr=classification_report(test_lab,np.argmax(np.round(predictions), axis=1),target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)


#Import data from Data_Loading
number_epochs,test_lab,class_types,batch_size,train_set_conv,train_im,valid_im,valid_set_conv,test_im,test_lab_categorical= Data_Loading.number_epochs, Data_Loading.test_labels, Data_Loading.class_types, Data_Loading.batch_size, Data_Loading.train_set_conv, Data_Loading.train_img, Data_Loading.valid_im, Data_Loading.valid_set_conv, Data_Loading.test_img, Data_Loading.test_lab_categorical


#-- ResNet 50 --
resnet50_model = ResNet50.resnet50()
#resnet50_model.summary()
resnet50_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['acc'])
batch=batch_size
nb_epochs=number_epochs
resnet_train = resnet50_model.fit(train_set_conv,epochs=nb_epochs,steps_per_epoch=train_im.shape[0]/batch,validation_steps=valid_im.shape[0]/batch,validation_data=valid_set_conv,callbacks=[ResNet50.lrdecay])

#Train and Validation evolution
loss = resnet_train.history['loss']
v_loss = resnet_train.history['val_loss']
acc = resnet_train.history['acc']
v_acc = resnet_train.history['val_acc']
epochs = range(len(loss))
#Plot
fig = plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.yscale('log')
plt.plot(epochs, loss, linestyle='--', linewidth=3, color='blue', alpha=0.6, label='Train Loss')
plt.plot(epochs, v_loss, linestyle='dotted', linewidth=2, color='green', alpha=0.8, label='Valid Loss')
plt.ylim(0.3, 100)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(fontsize=10)
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, linestyle='--', linewidth=3, color='blue', alpha=0.6, label='Train Accuracy')
plt.plot(epochs, v_acc, linestyle='dotted', linewidth=2, color='green', alpha=0.8, label='Valid Accuracy')
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

#Confusion matrix
pred_class_resnet50 = resnet50_model.predict(test_im)
conf_matrix(pred_class_resnet50)


#-- TEST --
test_result = resnet50_model.evaluate(test_im, test_lab_categorical, verbose=0)
print ("ResNet50 test loss: ", test_result[0])
print ("ResNet50 test accuracy: ", test_result[1])