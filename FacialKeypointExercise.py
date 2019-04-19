from KeypointHelper import Preprocess
from ambrose_flow import DataLoader
from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Training and Test Keypoints path 
csv_path = "data/training_frames_keypoints.csv"
csv_path_test = "data/test_frames_keypoints.csv"
# Training, and Test image dataset path
image_path = "data/training/"
image_path_test = "data/test/"
# HaarCascade path
cas_path = "data/detector_architectures/haarcascade_frontalface_default.xml"
# Validation set on my face...
random_paths = "data/valid/"
dims = 96
def Model(dims=96,batch_size=16,epochs=5):
      '''
      Fuction that runs Vgg16 architecture until it's optimize
      '''
      # Obtaining the datasets
      def getDataset():
            '''
            importing in the images from folder
            importing keypoints from csv
            cropping the image, and keypoints
            looping to obtain all images..
            '''
            p = Preprocess(csv_path, image_path)
            img_l = []
            key_l = []
            for i in range(p.getNumImage()):
                  image_new, keypts_new = p.Resize_data(dims,i)
                  image_new = p.Grayscale(p.Crop(image_new,is_image=True,crop=(dims,dims)))
                  img_l.append(image_new)
                  key_l.append(keypts_new)
            return [np.array(img_l),np.array(key_l)]
      def root_mean_squared_error(y_true, y_pred):
            '''
            Loss function
            '''
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

      def Preprocess_data():
            '''
            Importing in the data, and normalizing the inputs..
            '''
            # import the data
            inputs, keypoints = getDataset()
            inputs_test,keypoints_test = getDataset()
            # expanding dims
            inputs = np.expand_dims(inputs, axis = 3).astype(np.float32)
            inputs_test = np.expand_dims(inputs_test, axis = 3)
            # reshaping the keypoint data
            keypoints = keypoints.reshape(keypoints.shape[0],-1).astype(np.float32)
            keypoints_test = keypoints_test.reshape(keypoints_test.shape[0],-1).astype(np.float32)
            # Scaling the inputs
            inputs = inputs/255.
            inputs_test = inputs/255.
            # Scaling the keypoints
            keypoints = (keypoints-100.)/50
            return inputs,inputs_test, keypoints, keypoints_test


      def Architecture():
            '''
            This function is a vgg16-modified architecture
            added dropout on the linear layers to prevent overfitting..
            '''
            # Building the Architecture
            model = Sequential()
            model.add(Conv2D(32,(3,3), input_shape =[dims,dims,1],activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(64,(3,3),activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(128, (3,3),activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Conv2D(256,(3,3), activation="relu"))
            model.add(MaxPool2D(2,2))
            model.add(Flatten())
            model.add(Dense(512, activation="relu"))
            model.add((Dropout(.1)))
            model.add(Dense(1024, activation="relu"))
            model.add((Dropout(.25)))
            model.add(Dense(136,activation=None))
            model.summary()
            return model
      def Train():
            '''
            Training the model..
            I'm using a costume loss function.
            RMSE
            '''
            inputs, inputs_test, keypoints,keypoints_test = Preprocess_data()
            model = Architecture()
            model.compile(loss=root_mean_squared_error, optimizer="adam", metrics=["accuracy"])
            checkpoint = ModelCheckpoint(image_path+"new.model.best.hdf5",verbose=1,save_best_only=True)
            hist = model.fit(inputs,keypoints,batch_size=batch_size,epochs=epochs,callbacks=[checkpoint],verbose=1,shuffle=True,validation_split=.1)
            score = model.evaluate(inputs_test,keypoints_test, verbose = 0)
            accuracy = 100*score[1]
            return model,hist,accuracy
      model,hist,accuracy = Train()
      return model, hist, accuracy
model,hist,accuracy = Model(epochs = 55)

def faceCascade(cas_path,index):
      p = 9
      image = TestData(random_paths,dims,index)
      face_cascade = cv2.CascadeClassifier(cas_path)
      faces = face_cascade.detectMultiScale(image,1.2,1)
      image_w_detect = image.copy()
      for (x,y,w,h) in faces:
            cv2.rectangle(image_w_detect,(x,y),(x+w,y+h),(255,0,0),1)
      plt.imshow(image_w_detect)
      return image[y-p:y+h+p,x-p:x+w+p]

def Plot_results(model,hist,accuracy):
      '''
      plotting the results..
      '''
      plt.plot(hist.history['loss'])
      plt.plot(hist.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      print("The results accuracy of theis model is {}".format(accuracy))
Plot_results(model,hist,accuracy)

def TestData(rand_path,dims,index):
      '''
      importing in data for testing
      resizing the image
      making the img into an nxnx1 array
      '''
      image_,_ = DataLoader(rand_path,resize_image=True,dims=(dims,dims)).import_image()
      plt.imshow(image_[index])
      gray_image = cv2.cvtColor(image_[index], cv2.COLOR_RGB2GRAY)
      plt.imshow(gray_image, cmap="gray")
      return gray_image

def Predict_(index,dims=96):
      '''
      predicting an image, index will index into the dataset for an image
      '''
      # Importing in show keypoints
      pp = Preprocess(csv_path, image_path)
      # Importing in test data
      roi = faceCascade(cas_path,index)
      print(roi.shape)
      image = cv2.resize(roi,(96,96))
      #reshaping the shape
      image = image.reshape(1,image.shape[0],image.shape[1],1)
      #loading the weights
      model.load_weights(image_path+"new.model.best.hdf5")
      #predicting the model
      pred = model.predict(image/255.)
      #reshaping the prediction
      pred = pred.reshape(pred.shape[0],68,-1)*50+100
      # showing results
      pp.show_keypoints(image.squeeze(),pred[0])
      return pred,image,roi
pred,image,roi = Predict_(2)