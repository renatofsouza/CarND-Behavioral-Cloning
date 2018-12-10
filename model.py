import csv
import cv2
import numpy as np
#from keras.models import Sequential
#from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2#, activity_l2
from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint, Callback

import matplotlib.pyplot as plt

# Reads image files and measurements. Returns numpy arrays of images and measurements
def read_data(path, augment_data=False):
    
    #open the CSV file containing the driving data log
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    #Load images and measurements
    images = []
    measurements = []

    for line in lines:
        # skip it if ~0 speed - not representative of driving behavior
        if float(line[6]) < 0.1 :
            continue

        for i in range(3): #read all three cameras 0=center, 1=left, 2=right 
            source_path = line[i]
            #filename = source_path.split("\\")[-1]
            filename = source_path.split("/")[-1]
            image_path = "data/IMG/" + filename
            image = preprocess_image(cv2.imread(image_path))
            #imgplot = plt.imshow(image)
            #plt.show()
            images.append(image)
            measurement = float(line[3])
            if i == 1: # left camera
                measurement += 0.25 
            elif i == 2: #
                measurement -= 0.25    
            measurements.append(measurement)
            
            # flip horizontally and invert steer angle, if magnitude is > 0.33
            if augment_data == True and abs(measurement) > 0.33:    
                images.append(cv2.flip(image, 1)) # Augment data by flipping the image
                measurements.append(measurement * -1.0) # Augment data by flipping the angle

    return np.array(images), np.array(measurements)

def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are 
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    #new_img = img[35:140,:,:]
    # crop to 40x320x3
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # scale to ?x?x3
    #new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img


def train_model_simple(X_train, y_train):
    model = Sequential()
    #model.add(Lambda(lambda x : x/255.0 -0.5, input_shape=(160,320,3)))
    #model.add(BatchNormalization(input_shape=INPUT_DIMENSIONS, axis=1))
    model.add(Flatten( input_shape=(160,320,3)))
    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit(X_train, y_train,validation_split=0.2, shuffle= True,epochs=3 )
    model.save('model.h5')


def train_model(X_train, y_train):


    model = Sequential()
    
    # Normalize
    #model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
    #cropping to remove sky and hood of the car
    #model.add(Cropping2D(cropping=((70,25),(0,0)))) 
    
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add a fully connected output layer
    model.add(Dense(1))

    # Compile and train the model, 
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.fit(X_train, y_train,validation_split=0.2, shuffle= True,epochs=6 )
    model.save('model.h5')


X_train, y_train = read_data("data/driving_log.csv", True)

print(X_train.shape)
print(y_train.shape)


train_model(X_train, y_train)

#TODO
# 1 Try LeNet
# Crop images
# Use images from multiple cameras
