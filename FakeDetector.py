from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Activation, Flatten
from keras.regularizers import l2


def ConvNet():

    net = Sequential(name='ConvNet')
    net.add(Dense(units=500, input_shape=(150, 150, 3), activation='relu', name='DenseIn'))
    for idx in range(1, 3):
        net.add(Conv2D(filters=int(128 / idx), kernel_size=(2, 2), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01), name=('LoopConv1_' + str(idx))))
        net.add(Conv2D(filters=int(64 / idx), kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01), name=('LoopConv2_' + str(idx))))
        net.add(Activation('relu', name=('ReluAct' + str(idx * 1))))
        net.add(Conv2D(filters=int(64 / idx), kernel_size=(5, 5), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01), name=('LoopConv3_' + str(idx))))
        net.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=('MaxPool' + str(idx * 1))))
        net.add(Dense(units=150, activation='relu', name=('DenseOut' + str(idx * 1))))
    net.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', kernel_regularizer=l2(0.01), name='ConvAdded'))
    net.add(Activation('relu'))
    net.add(Conv2D(filters=1, kernel_size=(2, 2), padding='same', kernel_regularizer=l2(0.01), name='ConvFlatAdded',
                   activation='relu'))
    net.add(Flatten())
    # net.add(GlobalMaxPooling2D(name='GlobalAvgPoolAdded'))
    # net.add(Dropout(0.2, name='DropoutAdded'))
    net.add(Dense(units=1000, activation='relu', name='Dense12Added'))
    net.add(Dense(units=250, activation='relu', name='Dense2Added'))
    net.add(Dense(units=125, activation='relu', name='Dense3Added'))
    net.add(Dense(units=50, activation='relu', name='Dense4Added'))
    net.add(Dense(units=12, activation='relu', name='Dense5Added'))
    net.add(Dense(units=1, activation='sigmoid', name='DenseOutAdded'))


    return net
