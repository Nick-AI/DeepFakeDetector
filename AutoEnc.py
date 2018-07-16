from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, UpSampling2D
from keras.regularizers import l2


def ConvNet():

    net = Sequential(name='ConvNet')
    net.add(Dense(units=500, input_shape=(None, None, 3), activation='relu', name='DenseIn'))
    for idx in range(1, 3):
        net.add(Conv2D(filters=int(128/idx), kernel_size=(2, 2), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01), name=('LoopConv1_'+str(idx))))
        net.add(Conv2D(filters=int(64/idx), kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01), name=('LoopConv2_'+str(idx))))
        net.add(Activation('relu', name=('ReluAct'+str(idx*1))))
        net.add(Conv2D(filters=int(64/idx), kernel_size=(5, 5), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01), name=('LoopConv3_'+str(idx))))
        net.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=('MaxPool'+str(idx*1))))
        net.add(Dense(units=150, activation='relu', name=('DenseOut'+str(idx*1))))
    # encoder end
    for idx in range(2, 0, -1):
        net.add(Dense(units=150, activation='relu'))
        net.add(UpSampling2D(size=(2, 2)))
        net.add(Conv2D(filters=int(64/idx), kernel_size=(5, 5), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01)))
        net.add(Activation('relu'))
        net.add(Conv2D(filters=int(64/idx), kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01)))
        net.add(Conv2D(filters=int(128/idx), kernel_size=(2, 2), padding='same', kernel_initializer='he_uniform',
                       kernel_regularizer=l2(0.01)))
    net.add(Dense(units=500, activation='relu'))
    net.add(Conv2D(filters=3, kernel_size=(2, 2), activation='relu', padding='same'))

    return net