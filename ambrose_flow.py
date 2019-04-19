import numpy as np
import imageio
import glob
import os
from scipy.misc import imresize
'''
The DataLoader  is used to import images, and classes.
'''
class DataLoader(object):
    '''
    this class loads input data from images..
    if one wants class types add class name at the beginning of an image
    example: 0_cat.jpg or 1_dog.jpg
    
    the images keeps aspect ratio
    
    '''
    def __init__(self,path_,resize_image = False, dims = (None,None)):
        self.path_ = path_
        self.resize_image = resize_image
        self.dims = dims
    def import_image(self):
        '''
        this module imports the images, if resize ==True it will resize to designated
        dimensions 
        '''
        # looking for jpg files
        jpgfiles_end = len([name for name in os.listdir(self.path_) if name.endswith(".jpg")])
        # grabbing the names of the files.. is the directory..
        names_ = [name for name in os.listdir(self.path_) if name.endswith(".jpg")]
        print(jpgfiles_end)
        try:
            if os.path.exists(self.path_) == True:
                print("Importing in images...")
                label = []
                dataset = []
                for i in range(jpgfiles_end):
                    paths = glob.glob(self.path_+names_[i])[0]
                    filename = names_[i]
                    label.append(int(filename[0]))
                    data = imageio.imread(paths)
                    # resizing the image data
                    if self.resize_image == True:
                          w,h,c = data.shape
                          if h > w:
                                new_h, new_w = (int(self.dims[0]*h/w),self.dims[0])
                          elif h < w:
                                new_h, new_w = (self.dims[0],int(self.dims[0]*w/h))
                          else:
                                new_h, new_w = (self.dims[0],self.dims[0])
                          data = imresize(data,  size= (new_w,new_h))
                          data = data[:self.dims[0],:self.dims[1]]
                          dataset.append(data)
                    else:
                          dataset.append(data)
                print("\n\tImage import is complete!")
            return np.array(dataset), np.array(label)
        except:
            print("Path incorrect or do not exists. \nPlease check the file path!")
            return None