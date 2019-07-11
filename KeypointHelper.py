import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import cv2
from scipy import misc


class Face_keypointLoader(object):
    '''
    this class is made for the easy of use of getting number of images,
    and keypoints getting the keypoints and images
    also image/keypoint paths
    '''
    def __init__(self, csv_path=None, img_path=None):
        self.csv_path = csv_path
        self.img_path = img_path
        self.data = pd.read_csv(self.csv_path)

    def getImageName(self, n):
        return self.data.iloc[n, 0]

    def getImagePath(self, n):
        return self.img_path + self.getImageName(n)

    def getNumImage(self,):
        return len(self.data.iloc[:, 0])

    def get_keypoints(self, n):
        keypoints = self.data.iloc[n, 1:].astype("float").values
        keypoints_rev = keypoints.reshape(keypoints.shape[0]//2, 2)
        return np.float32(keypoints_rev)

    def get_image(self, n):
        image = np.float32(imageio.imread(self.img_path+self.getImageName(n)))
        return image[:, :, :3]

    def show_keypoints(self, image, key_pts):
        """
        Show image with keypoints
        """
        plt.imshow(image, cmap="gray")
        plt.scatter(key_pts[:, 0], key_pts[:, 1], s=10, marker='.', c='m')


class Preprocess(Face_keypointLoader):
    '''
    this class is design to preprocess the data as in cropping or resizing
    '''

    def __init__(self, csv_path=None, img_path=None):
        Face_keypointLoader.__init__(self, csv_path, img_path)
        self.csv_path = csv_path
        self.img_path = img_path
        self.data = pd.read_csv(self.csv_path)

    def Resize_data(self, size, n):
        image = self.get_image(n)
        w, h, c = image.shape
        if h > w:
            new_h, new_w = (int(size*h/w), size)
        elif h < w:
            new_h, new_w = (size, int(size*w/h))
        else:
            new_h, new_w = (size, size)
        keypoints = self.get_keypoints(n) * [new_w/w, new_h/h]
        return misc.imresize(image, (new_w, new_h)), keypoints

    def Crop(self, input_, crop=(None, None),
             is_image=False, is_keypoint=False):
        if is_image:
            x = input_[:crop[0], :crop[1]]
            return x
        elif is_keypoint:
            y = input_
            return y

    def Grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
