from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv2D,Dense,Dropout,)
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
Image_size=[224,224]
train_path=('F:/cabbage project/cabbage leaf dataset/train')
valid_path=('F:/cabbage project/cabbage leaf dataset/validation')
train_datagenerator=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,vertical_flip=True,rotation_range=20,width_shift_range=0.2)
test_datagenerator=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,rotation_range=20,width_shift_range=0.2)
valid_datagenerator=ImageDataGenerator(rescale=1/255,shear_range=0.2,zoom_range=0.5,horizontal_flip=True,rotation_range=20,width_shift_range=0.2)
test_dataset=test_datagenerator.flow_from_directory('F:/cabbage project/cabbage leaf dataset/test',target_size=(224,224),batch_size=32,seed=4,class_mode='categorical')
train_dataset=train_datagenerator.flow_from_directory('F:/cabbage project/cabbage leaf dataset/train',target_size=(224,224),batch_size=32,seed=4,class_mode='categorical')
valid_dataset=valid_datagenerator.flow_from_directory('F:/cabbage project/cabbage leaf dataset/validation',target_size=(224,224),batch_size=32,seed=4,class_mode='categorical')
vgg16=VGG16(input_shape=Image_size+[3],weights='imagenet',include_top=False)
for layer in vgg16.layers:
    layer.trainable=False
number_of_classes=glob('F:/cabbage project/cabbage leaf dataset/train/*')
flatten_layer=Flatten()(vgg16.output)
output_layer=Dense(len(number_of_classes),activation='softmax')(flatten_layer)
vgg16_model=Model(inputs=vgg16.input,outputs=output_layer)
vgg16_model.summary()
vgg16_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=vgg16_model.fit(train_dataset,validation_data=valid_dataset,epochs=40,steps_per_epoch=61)
vgg16_model.save("F:/cabbage project/vgg16.h5")
vgg16_model.load_model("F:/cabbage project/vgg16.h5")
