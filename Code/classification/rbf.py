import numpy as np
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Reshape, Conv2D, Flatten, Layer
# from keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from preprocess.utils import *

np.random.seed(42)  # 随机数种子


class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        self.units = units
        self.gamma = gamma
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[1]),
                                       initializer='uniform',
                                       trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.centers
        l2 = K.sqrt(K.sum(K.pow(diff, 2), axis=1))
        output = K.exp(-self.gamma * K.pow(l2, 2))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


if __name__ == '__main__':
    x, y, label = load_data()

    y = to_categorical(y, num_classes=8)  # y=[0-7]
    trainX, trainY, testX, testY = train_test_split(x, y)

    smote = SMOTE(random_state=4487)
    trainX, trainY = smote.fit_resample(trainX, trainY)

    testset = (testX, testY)

    K.clear_session()  # cleanup
    random.seed(4487)
    tf.random.set_seed(4487)  # initialize seed

    # build the network
    rnn = Sequential()
    rnn.add(RBFLayer(units=x.shape[1], gamma=0.1, input_shape=(x.shape[1],)))  # 添加RBF层
    rnn.add(Dense(units=64, activation='relu'))  # 添加全连接层
    rnn.add(Dense(units=8, activation='softmax'))  # 添加输出层

    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # use validation accuracy for stopping
        # (use 'val_acc' for tf1)
        min_delta=0.00001, patience=5,
        verbose=1, mode='auto'
    )

    callbacks_list = [earlystop]

    # compile and fit the network
    rnn.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True),
                metrics=['accuracy'])
    history = rnn.fit(trainX, trainY, epochs=100, batch_size=50,
                      callbacks=callbacks_list,
                      validation_data=testset, verbose=1)
