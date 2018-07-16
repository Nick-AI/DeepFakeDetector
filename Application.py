import sys
import os
import pickle
import FrameGetter
import ModelDriver
import DeepFakeSequence
import numpy as np
from shutil import copyfile
from skimage.io import imread
from skimage.transform import resize

if __name__ == '__main__':

    vid_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    pred_dir = dest_dir + '/imagesToReview'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    batch_size = 32
    susp_count = 0
    conv = FrameGetter.VideoConverter(dest_dir)
    conv.extract_faces_from_vid(vid_dir)
    dataset = conv.convert_to_pickle('pickleStorage')
    data = pickle.load(open(dataset, "rb"))
    data = [[k, v] for k, v in data.items()]

    classifier = ModelDriver.get_tester()
    for idx in range(int(np.floor(len(data) / batch_size))):
        batch_x = [data[i][0] for i in range(idx * batch_size, (idx + 1) * batch_size)]
        out_x = []
        for item in batch_x:
            pic = np.expand_dims(np.array(resize(imread(item), (150, 150))), 0)
            pred = classifier.predict(pic, batch_size=1)
            print(pred[0])
            if pred[0][0] < 1.0e-6:  # 1.0e-8 for cage
                copyfile(item, pred_dir + '/' + str(susp_count) + '.jpg')
                susp_count += 1
    print('0')