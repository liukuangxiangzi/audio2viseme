
#prprocess Y

import os
from datetime import datetime
import numpy as np
import argparse
from keras.layers import Input, LSTM
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import *
from keras.utils import plot_model



#-x path-of-audio-feature-file-for-training, -X path-of-audio-feature-file-for-testing, -y path-of-viseme-file-for-training -Y path-of-viseme-file-for-testing -o path-of-saving-model
#e.g. python train.py -x data/audio_feature_train.npy -X data/audio_feature_test.npy -y data/train_viseme_13.npy -Y data/test_viseme_13.npy -o model/audio2pho_model_mfa13label_ep300_1e-4_32.h5


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-x", "--train_x", type=str, help="input audio feature file for training")
parser.add_argument("-X", "--test_X", type=str, help="input audio feature file for testing")
parser.add_argument("-y", "--train_y", type=str, help="input viseme file for training")
parser.add_argument("-Y", "--test_Y", type=str, help="input viseme file for testing")
parser.add_argument("-o", "--out_fold", type=str, help="model folder")
args = parser.parse_args()



def labelcategorical():
    label_train = np.array(np.load(args.train_y))  #train_viseme_label14H.npy
    label_test = np.array(np.load(args.test_Y))

    label_train = label_train.reshape(753,75,-1)
    label_test = label_test.reshape(191,75,-1)

    # one hot encode y
    label_categorical_train = to_categorical(label_train)
    label_categorical_test = to_categorical(label_test)
    return label_categorical_train, label_categorical_test


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all x
    trainX = np.load(args.train_x)  #trainX.shape = (753, 75, 768)
    print('trainX.shape', trainX.shape)
    testX = np.load(args.test_X)   #testX.shape = (191, 75, 768)
    print('testX.shape', testX.shape)
    # load all y
    trainy, testy = labelcategorical()
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


#set tensorboard
#tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/audio2viseme/logs/ --host=127.0.0.1
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tbCallBack = TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=True)

#X y data
trainX, trainy, testX, testy = load_dataset(prefix='')

verbose, epochs, batch_size = 0, 300, 32
h_dim = 256
n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
print(n_timesteps, n_features)

drpRate = 0.2
recDrpRate = 0.2
lr = 1e-4
initializer = 'glorot_uniform'

# define model
net_in = Input(shape=(n_timesteps, n_features))
lstm1 = LSTM(h_dim,
             activation='sigmoid',
             dropout=drpRate,
             recurrent_dropout=recDrpRate,
             return_sequences=True)(net_in)
lstm2 = LSTM(h_dim,
             activation='sigmoid',
             dropout=drpRate,
             recurrent_dropout=recDrpRate,
             return_sequences=True)(lstm1)
lstm3= LSTM(h_dim,
            activation='sigmoid',
            dropout=drpRate,
            recurrent_dropout=recDrpRate,
            return_sequences=True)(lstm2)

dropout = Dropout(0.5)(lstm3)

l1 = Dense(128,
           kernel_initializer=initializer,
           name='lm_Dense1',activation='relu')(dropout)

out = Dense(13,
            kernel_initializer=initializer, name='lm_out',activation='softmax')(l1)

model = Model(inputs=net_in, outputs=out)
model.summary()
opt = Adam(lr=lr)
#opt = SGD(learning_rate=lr)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])



plot_model(model, to_file='audio2viseme_model.png', show_shapes=True, show_layer_names=True)

model.fit(trainX, trainy,
          validation_data=(testX, testy),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[tbCallBack]
          )

model.save(args.out_fold)
print("Saved model to disk")










