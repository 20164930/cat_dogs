import os  # 处理字符串路径
import glob  # 查找文件

from keras.layers import BatchNormalization
from keras.models import Sequential  # 导入Sequential模型
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import metrics
import numpy as np
from pandas import Series, DataFrame
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
# 加载数据
import os
from PIL import Image
import PIL
import numpy as np
from keras.preprocessing import image
import random
from keras.callbacks import ModelCheckpoint

def load_data(path):
    files = os.listdir(path)
    random.shuffle(files)
    images = []
    labels = []
    for f in files:
        img_path = path + '/' + f
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        images.append(img_array)
        if 'cat' in f:
            labels.append(0)  # 0是猫
        else:
            labels.append(1)

    datas = np.array(images)
    labels = np.array(labels)
    label = np_utils.to_categorical(labels, 2)
    return datas, label


data, label = load_data('tr')
print(data.shape)
train_data = data[:2000]
train_labels = label[:2000]
validation_data = data[2000:2500]
validation_labels = label[2000:2500]

model = Sequential()
# Block 1, 2层
model.add(Convolution2D(16, 3, 3, activation='relu',
                        border_mode='same', input_shape=(224, 224,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 2, 2层
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Block 3, 3层
#model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Block 4, 4层
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# Classification block, 全连接3层
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))


checkpointer = ModelCheckpoint(filepath="checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5",
save_best_only=False, verbose=1,  period=20)

model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
model.fit(train_data, train_labels,
         nb_epoch=500, batch_size=100,
         validation_data=(validation_data, validation_labels),callbacks=[checkpointer])


