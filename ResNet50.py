import Data_Loading
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2


def res_id(x, filters):
  #Dimension does not change
  x_skip = x
  f1, f2 = filters

  #1st block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #2nd block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #3rd block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x

def res_conv(x, s, filters):
  #here the input size changes
  x_skip = x
  f1, f2 = filters

  #1st block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map maybe delete later
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #2nd block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #3rd block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)

  #Shortcut
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  #Add
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x

def resnet50():
  input_im = Input(shape=(Data_Loading.train_img.shape[1], Data_Loading.train_img.shape[2], Data_Loading.train_img.shape[3])) # cifar 10 images size
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  #1st stage, contains maxpooling
  x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage, from here there is no more pooling
  x = res_conv(x, s=1, filters=(64, 256))
  x = res_id(x, filters=(64, 256))
  x = res_id(x, filters=(64, 256))

  #3rd stage
  x = res_conv(x, s=2, filters=(128, 512))
  x = res_id(x, filters=(128, 512))
  x = res_id(x, filters=(128, 512))
  x = res_id(x, filters=(128, 512))

  #4th stage
  x = res_conv(x, s=2, filters=(256, 1024))#TODO Modify s to 1 to remove down-sampling ?
  x = res_id(x, filters=(256, 1024))
  x = res_id(x, filters=(256, 1024))
  x = res_id(x, filters=(256, 1024))
  x = res_id(x, filters=(256, 1024))
  x = res_id(x, filters=(256, 1024))

  #5th stage
  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_id(x, filters=(512, 2048))
  x = res_id(x, filters=(512, 2048))

  #Average pooling and Dense connection
  x = AveragePooling2D((2, 2), padding='same')(x)
  x = Flatten()(x)
  x = Dense(len(Data_Loading.class_types), activation='softmax', kernel_initializer='he_normal')(x) #multi-class

  #Model
  model = Model(inputs=input_im, outputs=x, name='Resnet50')
  return model

#-- Callbacks --
def lrdecay(epoch):
    lr = 1e-4
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay

def earlystop(mode):
  if mode=='acc':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
  elif mode=='loss':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
  return estop