import random
from collections import Counter
import tensorflow as tf
# from tensorflow.keras.optimizer_experimental.adam import Adam
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

from preprocess.utils import *

if __name__ == '__main__':
    x, y, label = load_data()
    y = to_categorical(y, num_classes=8)  # y=[0-7]
    trainX, trainY, testX, testY = train_test_split(x, y)
    # print('Original dataset shape %s' % Counter(trainY))

    smote = SMOTE(random_state=4487)
    trainX, trainY = smote.fit_resample(trainX, trainY)
    # print('Original dataset shape %s' % Counter(trainY))

    testset = (testX, testY)

    K.clear_session()  # cleanup
    random.seed(4487)
    tf.random.set_seed(4487)  # initialize seed

    # build the network
    nn = Sequential()
    nn.add(Dense(units=30, input_dim=x.shape[1], activation='relu'))
    # nn.add(Dense(units=30, activation='relu'))
    nn.add(Dense(units=20, activation='relu'))
    nn.add(Dense(units=10, activation='relu'))
    # nn.add(Dropout(rate=0.9, seed=44))
    nn.add(Dense(units=8, activation='softmax'))

    # early stopping criteria
    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # use validation accuracy for stopping
        # (use 'val_acc' for tf1)
        min_delta=0.00001, patience=5,
        verbose=1, mode='auto'
    )

    callbacks_list = [earlystop]
    optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, nesterov=True)
    # optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)

    # compile and fit the network
    nn.compile(loss=keras.losses.categorical_crossentropy,
               optimizer=optimizer,
               metrics=['accuracy']  # also calculate accuracy during training
               )

    history = nn.fit(trainX, trainY, epochs=100, batch_size=50,
                     callbacks=callbacks_list,
                     validation_data=testset,  # specify the validation set
                     verbose=1)
