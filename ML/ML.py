import sys
import json
import cv2
class Dataset:
    def __init__(self, filename, split=0.8, traincount=300):
        self.filename = filename
        self.split = split
        if(self.split > 0.9):
            print("Invalid split!")
            sys.exit(1)
        self.traincount = traincount    
        
        with open(self.filename) as jsonfile:
            self.data = json.load(jsonfile)
    
    def save(self):
        with open(self.filename, "w") as jsonfile:
            json.dump(self.data, jsonfile)
    
    def parse(self):
        print("Parsing data for neural net...")
        
        if(len(self.data) > self.traincount):
            print("Getting next " + str(self.traincount) + " images...")
        else:
            print("Getting all " + str(len(self.data)) + " remaining images with data.")
        
        X = []
        y = []
        i = 0
        drop = []
        
        for img in self.data:
            if(i == self.traincount):
                break
            #imgData = image.load_img("data-img/" + img)
            imgData = cv2.imread("data-img/" + img)
            #imgData = image.img_to_array(imgData)
            X.append(imgData)
            y.append(np.array(self.data[img]))
            drop.append(img)
            i += 1
        
        for img in drop:
            self.data.pop(img)
            
        # train_test_split was giving me problems, luckily it's easy as hell to write your own
        train_count = int(len(X) * self.split)
        test_count = int(len(X) - train_count)
        for i in range(0, len(X)):
            X[i] = X[i][None, ...]
            y[i] = y[i][None, ...]
        
        X_train = np.array(X[:-test_count])
        X_valid = np.array(X[train_count:])
        y_train = np.array(y[:-test_count])
        y_valid = np.array(y[train_count:])
        print("Got " + str(len(X_train)) + " images for training and " + str(len(X_valid)) + " images for validation.")
        return X_train, X_valid, y_train, y_valid
    
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Reshape
#from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np

class NeuralNet:
    def __init__(self, dataset):
        self.dataset = dataset

        print("Looking for existing model...")
        try:
            self.loadModel("model.cm")
        except:
            print("Not found. Generating a new model...")
            if(len(dataset.data) > 1):
                X_train, X_valid, y_train, y_valid = self.dataset.parse()
                self.buildModel()
                self.trainModel(X_train, X_valid, y_train, y_valid)
                self.model.save("model.cm")
            else:
                print("Dataset empty! Can't generate model!")
    
    def loadModel(self, filename):
        self.model = load_model(filename)

    def buildModel(self):
        print("Building model...")
        self.model = Sequential()
        #self.model.add(Reshape((1, 540, 960, 3), input_shape=(540, 960, 3)))
        #self.model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(None, 540, 960, 3)))
        self.model.add(Lambda(lambda x: tf.cast(x, tf.float32)))
        self.model.add(Lambda(lambda x: x/127.5-1.0))
        self.model.add(Conv2D(24, 5, 5, activation='elu'))
        self.model.add(Conv2D(36, 5, 5, activation='elu'))
        self.model.add(Conv2D(48, 5, 5, activation='elu'))
        self.model.add(Conv2D(64, 3, 3, activation='elu'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='elu'))
        self.model.add(Dense(50, activation='elu'))
        self.model.add(Dense(10, activation='elu'))
        self.model.add(Dense(3))

    def train(self):
        while(len(self.dataset.data)):
            X_train, X_valid, y_train, y_valid = self.dataset.parse()
            self.trainModel(X_train, X_valid, y_train, y_valid)

    def trainModel(self, X_train, X_valid, y_train, y_valid):
        print("Training model, this may take a while...")
        checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val-loss', verbose = 0, save_best_only = True, mode = "auto")
        #self.model.compile(loss='mean_squared_error', optimizer=Adam())
        self.model.compile(loss='hinge', optimizer=Adam())
        print(X_train[0].shape)       
        self.model.fit(x=X_train, y=y_train, validation_data = (X_valid, y_valid), batch_size = 10, epochs = 100)
        self.dataset.data = {} # We don't need to train the neural net on the same data every time
        
    def predict(self, frame):
        #return self.model.predict(self.camera.getFrame()[None, ...])
        return self.model.predict(frame)
