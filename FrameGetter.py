import cv2
import os
import dlib
import random
import pickle
import numpy as np
from PIL import Image, ImageOps
from skimage import io


class VideoConverter:

    start_idx = 0
    video_dir = ''
    vid_name = ''
    is_fake = True
    TARGET_FOLDER = ''
    FRAME_FOLDER = ''
    FACE_FOLDER = ''
    FACE_DETECTOR = None

    def __init__(self, target_folder):
        self.TARGET_FOLDER = target_folder
        self.FACE_DETECTOR = dlib.get_frontal_face_detector()
        self.FRAME_FOLDER = target_folder + '/frames'
        self.FACE_FOLDER = target_folder + '/extracted_faces'

    def __convert_video(self, v_dir):
        """
        extracts frames from video

        :param v_dir: string
        :return:
        """
        self.video_dir = v_dir
        vid_capt = cv2.VideoCapture(v_dir)
        curr_frame = 0
        # clear directory if it already exists, else create it
        if os.path.exists(self.FRAME_FOLDER):
            for file in os.listdir(self.FRAME_FOLDER):
                file_path = os.path.join(self.FRAME_FOLDER, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
        else:
            os.makedirs(self.FRAME_FOLDER)

        while True:
            # ret is return value, once it turns False, video is over
            ret, frame = vid_capt.read()
            if not ret:
                break
            f_name = self.FRAME_FOLDER + '/' + self.vid_name + 'frame' + str(curr_frame) + '.jpg'
            cv2.imwrite(f_name, frame)
            curr_frame += 1

        vid_capt.release()
        cv2.destroyAllWindows()

    def __extract_face(self, frame_file, resize_to, frame_idx):
        """
        :param frame_dir: frame filename
        :param resize_to: tuple of dimensions, e.g. 300,300
        :return:
        """
        image = io.imread(self.FRAME_FOLDER + '/' + frame_file)
        det_faces = self.FACE_DETECTOR(image, 1)
        face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in det_faces]
        # Crop faces and plot
        for n, face_rect in enumerate(face_frames):
            try:
                face = Image.fromarray(image).crop(face_rect)
                face = ImageOps.fit(face, resize_to, Image.ANTIALIAS)
                if not os.path.exists(self.FACE_FOLDER):
                    os.makedirs(self.FACE_FOLDER)
                if self.is_fake:
                    safe_dir = self.FACE_FOLDER + '/' + self.vid_name + '_face' + str(frame_idx) + '_0' +'.jpg'
                else:
                    safe_dir = self.FACE_FOLDER + '/' + self.vid_name + '_face' + str(frame_idx) + '_1' +'.jpg'
                face.save(safe_dir, 'JPEG')
            except IOError as e:
                print('There was an issue with creating a thumbnail for this frame')
                print(e)

    def __get_face_frames(self, folder_dir=None):
        if folder_dir is not None:
            self.FRAME_FOLDER = folder_dir
        frame_dir = self.FRAME_FOLDER
        for idx, frame in enumerate(os.listdir(frame_dir)):
            self.__extract_face(frame, (200, 200), idx)

    def extract_faces_from_vid(self, vid_dir):
        self.vid_name = vid_dir.split('/')[-1].split('.')[0]
        if vid_dir.split('/')[-2] == 'Originals':
            self.is_fake = False
        self.__convert_video(vid_dir)
        self.__get_face_frames()
        return self.FACE_FOLDER

    def extract_faces_from_images(self, image_folder):
        self.__get_face_frames(image_folder)

    def __convert_image_to_RGB(self, image):
        img_data = np.asarray(image)
        image = Image.fromarray(np.roll(img_data, 1, axis=-1))
        return np.asarray(image, dtype='float64')

    def convert_to_LMDB_with_images(self, dest_folder):
        destination = self.TARGET_FOLDER + '/' + dest_folder
        db_writer = px.Writer(dirpath=destination, map_size_limit=15000, ram_gb_limit=15)
        images = []
        targets = []
        for frame in os.listdir(self.FACE_FOLDER):
            image = self.__convert_image_to_RGB(cv2.imread(self.FACE_FOLDER + '/' + frame))
            images.append(image)
            if '_1.jpg' not in frame:
                targets.append(np.asarray([1], dtype='int64'))
            else:
                targets.append(np.asarray([0], dtype='int64'))
        db_writer.put_samples('input', np.asarray(images), 'target', np.asarray(targets))

    # def convert_to_LMDB(self, dest_folder):
    #     fc = 0
    #     destination = self.FACE_FOLDER + '/' + dest_folder
    #     db_writer = px.Writer(dirpath=destination, map_size_limit=25000, ram_gb_limit=25)
    #     images = []
    #     targets = []
    #     for frame in os.listdir(self.FACE_FOLDER):
    #         if not frame == 'binStorage':
    #             if '_1.jpg' not in frame:
    #                 targets.append(np.asarray([1], dtype='int64'))
    #                 os.rename(self.FACE_FOLDER + '/' + frame, self.FACE_FOLDER + '/face' + str(fc) + '_0.jpg')
    #                 images.append(self.FACE_FOLDER + '/face' + str(fc) + '_0.jpg')
    #             else:
    #                 targets.append(np.asarray([0], dtype='int64'))
    #                 os.rename(self.FACE_FOLDER + '/' + frame, self.FACE_FOLDER + '/face' + str(fc) + '_1.jpg')
    #                 images.append(self.FACE_FOLDER + '/face' + str(fc) + '_1.jpg')
    #             fc += 1
    #     temp = list(zip(images, targets))
    #     random.shuffle(temp)
    #     images, targets = zip(*temp)
    #     db_writer.put_samples('input', np.asarray(images), 'target', np.asarray(targets))

    def convert_to_pickle(self, dest_folder):
        destination = self.FACE_FOLDER + '/' + dest_folder
        if not os.path.exists(destination):
            os.makedirs(destination)
        destination += '/storage.pkl'
        data = {}
        for frame in os.listdir(self.FACE_FOLDER):
            if '.' not in frame:
                pass
            elif '_1.jpg' not in frame:
                data[self.FACE_FOLDER + '/' + frame] = 0
            else:
                data[self.FACE_FOLDER + '/' + frame] = 1
        self.__pickle_save(data, destination)
        return destination

    @staticmethod
    def __pickle_save(data, destination_dir):
        handle = open(destination_dir, 'wb')
        pickle.dump(data, handle)
        return


if __name__ == '__main__':

    # vids = ['./Dataset/Originals/TrumpSpeech4.mp4']
    tester = VideoConverter('./Dataset/')
    # for video in vids:
    #     tester.extract_faces_from_vid(video)
    tester.convert_to_pickle('pickleStorage')

