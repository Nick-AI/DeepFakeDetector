from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
from PIL import Image
import numpy as np
import random
import pickle


class TrainSequence(Sequence):

    def __init__(self, storage_dir, batch_size):
        self.batch_size = batch_size
        # self.data = px.Reader(storage_dir)
        self.data = pickle.load(open(storage_dir + '/storage.pkl', "rb"))
        self.data = [[k, v] for k, v in self.data.items()]
        random.shuffle(self.data)

    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))

    def __getitem__(self, idx):
        batch_x = [self.data[i][0] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        batch_y = [self.data[i][1] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]

        out_x = []
        for item in batch_x:
            out_x.append(resize(imread(item), (150, 150)))
        return np.array(out_x), batch_y


class TestSequence(Sequence):

    def __init__(self, storage_dir, batch_size):
        self.batch_size = batch_size
        # self.data = px.Reader(storage_dir)
        self.data = pickle.load(open(storage_dir + '/storage.pkl', "rb"))
        self.data = [[k, v] for k, v in self.data.items()]
        random.shuffle(self.data)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = [self.data[i][0] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        batch_y = [self.data[i][1] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]

        out_x = []
        for item in batch_x:
            out_x.append(resize(imread(item), (150, 150)))
        return np.array(out_x), batch_y

class PredictSequence(Sequence):

    def __init__(self, storage_dir, batch_size):
        self.batch_size = batch_size
        # self.data = px.Reader(storage_dir)
        self.data = pickle.load(open(storage_dir, "rb"))
        self.data = [[k, v] for k, v in self.data.items()]

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = [self.data[i][0] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]

        out_x = []
        for item in batch_x:
            out_x.append(resize(imread(item), (150, 150)))
        return np.array(out_x)



class AutoEncSequence(Sequence):

    def __init__(self, storage_dir, batch_size):
        self.batch_size = batch_size
        self.data = pickle.load(open(storage_dir + '/storage.pkl', "rb"))
        self.data = [[k, v] for k, v in self.data.items()]
        random.shuffle(self.data)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = [self.data[i][0] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        batch_y = [self.data[i][1] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]

        out_x = []
        for item in batch_x:
            out_x.append(resize(imread(item), (100, 100)))
        out_x=np.array(out_x)
        return out_x, out_x
