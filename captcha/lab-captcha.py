#!/usr/bin/python3

from keras import backend as K
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, merge
import keras.metrics
from keras.models import Model, load_model
from keras.utils import np_utils
import os
import random
import numpy as np
import PIL
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

#importing math libraries
import matplotlib
import matplotlib.pyplot as plt

# Don't modify BEGIN

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

seed = 7
np.random.seed(seed)

# END

WIDTH = 160
HEIGHT = 60
CHANNEL = 3

BS = 32
EPOCHS = 32
def one_hot_encode (label) :
    return np_utils.to_categorical(np.int32(list(label)), 10)

def accuracy(test_labels, predict_labels):
    y1 = K.cast(K.equal(K.argmax(test_labels[:,0,:]), K.argmax(predict_labels[:,0,:])), K.floatx())
    y2 = K.cast(K.equal(K.argmax(test_labels[:,1,:]), K.argmax(predict_labels[:,1,:])), K.floatx())
    y3 = K.cast(K.equal(K.argmax(test_labels[:,2,:]), K.argmax(predict_labels[:,2,:])), K.floatx())
    y4 = K.cast(K.equal(K.argmax(test_labels[:,3,:]), K.argmax(predict_labels[:,3,:])), K.floatx())
    acc = K.mean(y1 * y2 * y3 * y4)
    return acc

def load_data(path,train_ratio):
    datas = []
    labels = []
    input_file = open(path + 'labels.txt')
    for i,line in enumerate(input_file):
        chal_img = PIL.Image.open(path + str(i) + ".png").resize((WIDTH, HEIGHT))
        data = np.array(chal_img).astype(np.float32)
        data = np.multiply(data, 1/255.0)
        data = np.asarray(data)
        datas.append(data)
        labels.append(one_hot_encode(line.strip()))
    input_file.close()
    datas_labels = list(zip(datas,labels))
    random.shuffle(datas_labels)
    (datas,labels) = list(zip(*datas_labels))
    size = len(labels)
    train_size = int(size * train_ratio)
    train_datas = np.stack(datas[ 0 : train_size ])
    test_datas = np.stack(datas[ train_size : size ])
    train_labels = np.stack(labels[ 0 : train_size ])
    test_labels = np.stack(labels[ train_size : size])
    return (train_datas,train_labels,test_datas,test_labels)

def get_cnn_net():
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
    
    x = Conv2D(32, (5, 5), padding='valid', input_shape=(HEIGHT, WIDTH, CHANNEL), activation='tanh')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
#   TODO!!!
    x = Conv2D(32, (5, 5), padding='valid', activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # x = Dropout(0.5)(x)

    x = Conv2D(64, (3, 3), padding='same',  activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # x = Dropout(0.3)(x)


    x = Conv2D(64, (3, 3), padding='same',  activation='tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.15)(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='tanh')(x)
    x = Dense(32, activation='sigmoid')(x)

    x1 = Dense(10, activation='softmax')(x)
    x2 = Dense(10, activation='softmax')(x)
    x3 = Dense(10, activation='softmax')(x)
    x4 = Dense(10, activation='softmax')(x)
    x = concatenate([x1,x2,x3,x4])
    x = Reshape((4,10))(x)
    model = Model(inputs=inputs, outputs=x)

    model.compile(loss='categorical_crossentropy', loss_weights=[1.], optimizer='Adam', metrics=[accuracy])
    return model
    
# Don't modify BEGIN

(train_datas,train_labels,test_datas,test_labels) = load_data('data/',0.9)
model = get_cnn_net()
print(model)

PATH = os.getcwd()
filename=PATH+'/model.('+str(HEIGHT)+'x'+str(WIDTH)+').'+str(len(train_datas))
checkpoint = ModelCheckpoint(filename+'.h5', monitor='val_accuracy', verbose=1, 
                            save_best_only=True, mode='max')

# check 5 epochs
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, mode='max') 
callbacks_list = [checkpoint, early_stop]

hist = model.fit(train_datas, train_labels, epochs=EPOCHS, batch_size=BS, verbose=1, validation_split=0.1, callbacks=callbacks_list)

#saving the best accuracy
model = load_model(filename+'.h5')
filename = filename+'.val_acc.('+str(round(hist.history['val_accuracy'][-6]*100, 2))+')'
model.save(filename+'.h5')

predict_labels = model.predict(test_datas,batch_size=BS)
test_size = len(test_labels)
y1 = test_labels[:,0,:].argmax(1) == predict_labels[:,0,:].argmax(1)
y2 = test_labels[:,1,:].argmax(1) == predict_labels[:,1,:].argmax(1)
y3 = test_labels[:,2,:].argmax(1) == predict_labels[:,2,:].argmax(1)
y4 = test_labels[:,3,:].argmax(1) == predict_labels[:,3,:].argmax(1)
acc = (y1 * y2 * y3 * y4).sum() * 1.0

acc = acc/test_size
print('\nmodel evaluate:\nacc:', acc)

y1_acc = (y1.sum()) *1.0/test_size
y2_acc = (y2.sum()) *1.0/test_size
y3_acc = (y3.sum()) *1.0/test_size
y4_acc = (y4.sum()) *1.0/test_size

print('y1', y1_acc)
print('y2', y2_acc)
print('y3', y3_acc)
print('y4', y4_acc)

# save results
with open('results.txt', 'a') as f:
    acc_str = '\n' +  str(acc) + ' ' + str(y1_acc) + ' ' + str(y2_acc) + ' ' + str(y3_acc) + ' ' + str(y4_acc)
    f.write(acc_str) 

# plot the training loss and accuracy
plt.style.use('ggplot')
N = len(hist.history['val_accuracy'])
plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(0, N), hist.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), hist.history['val_accuracy'], label='val_acc')
plt.title('Training Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')
plt.legend(loc='lower left')


# plt.figure(2)
plt.subplot(212)
plt.plot(np.arange(0, N), hist.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), hist.history['val_loss'], label='val_loss')
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend(loc='lower left')

plt.savefig(filename+'.png')

K.clear_session()
del sess

# END
