import FakeDetector
import DeepFakeSequence
import time
from tqdm import tqdm
import keras
import keras.backend as K
from keras.datasets import cifar10
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.callbacks import EarlyStopping, Callback

dir = "."

def get_tester():
    mod_dir = dir + '/Model/generalModel.h5'
    classifier = keras.models.load_model(mod_dir)
    return classifier


if __name__ == '__main__':
    batch_size = 32
    nb_epochs = 10
    train_dir = dir + '/Dataset/faces/Train/pickleStorage'
    test_dir = dir + '/Dataset/faces/Test/pickleStorage'
    model_dir = dir + '/Model/generalModel.json'
    train_gen = DeepFakeSequence.TrainSequence(train_dir, batch_size)
    test_gen = DeepFakeSequence.TestSequence(test_dir, batch_size)

    for i in range(10):
        classifier = FakeDetector.ConvNet()
        classifier.summary()
        learning_rate = 0.1
        opt = Adadelta(lr=learning_rate)
        early_stopper = EarlyStopping(monitor='acc', min_delta=0.01, patience=20)
        classifier.load_weights(dir + '/AutoEncModel/autoEnc_weights.h5', by_name=True)
        classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        start = time.clock()
        print(i, '\n')
        classifier.fit_generator(generator=train_gen, epochs=200, verbose=2, validation_data=test_gen,
                                        shuffle=True, callbacks=[early_stopper])
        if time.clock()-start > 750:
            print('WAS SAVED\nWAS SAVED')
            with open(model_dir, 'w') as json_dir:
                json_dir.write(classifier.to_json())
            classifier.save(model_dir.replace('.json', '.h5'))
            classifier.save_weights(model_dir.replace('spicerTrumpHopeDrEvilModel.json',
                                                      'spicerTrumpHopeDrEvilModel_weights.h5'))
