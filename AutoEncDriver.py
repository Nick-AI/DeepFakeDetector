import AutoEnc
import DeepFakeSequence
from tqdm import tqdm
import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam, Adadelta


if __name__ == '__main__':
    batch_size = 32
    nb_epochs = 10
    train_dir = './Dataset/faces/AutoEnc/pickleStorage'
    model_dir = './AutoEncModel/autoEnc.json'
    train_gen = DeepFakeSequence.AutoEncSequence(train_dir, batch_size)
    classifier = AutoEnc.ConvNet()
    classifier.summary()
    learning_rate = 0.01
    opt = Adadelta(lr=learning_rate)
    classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    for e in tqdm(range(nb_epochs)):
        if e == int(0.5 * nb_epochs):
            K.set_value(classifier.optimizer.lr, np.float32(learning_rate / 10.))
        if e == int(0.8 * nb_epochs):
            K.set_value(classifier.optimizer.lr, np.float32(learning_rate / 100.))

        classifier.fit_generator(generator=train_gen, epochs=5, verbose=1, shuffle=True)
    with open(model_dir, 'w') as json_dir:
        json_dir.write(classifier.to_json())
    classifier.save(model_dir.replace('.json', '.h5'))
    classifier.save_weights(model_dir.replace('autoEnc.json', 'autoEnc_weights.h5'))
