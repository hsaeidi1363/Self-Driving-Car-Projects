#importing useful packages
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import json
import cv2
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
tf.python.control_flow_ops = tf

print('Modules loaded.')

########some functions for preprocessing the track image before passing to the neural network
# converting the image to grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# applying sobel algorithm for finding gradients of the image along Y axis
def abs_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = grayscale(img)
    # 2) Take the derivative 
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) separate the Saturation and Light channels
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    # 3) Apply the thresholds to the channels and return a binary image of threshold result
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

################################################################################################
# reading the csv file that contains the image addresses as well as driving info (e.g. steering angle, throttle, etc.)
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		# making sure the first line is not considered as data
		if line[0] != 'center':
			samples.append(line)

print('The total number of images from the center camera: ', len(samples))
# Splitting 20% of the data for validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state = 1)

# data generator function for training and validation sets: In short, it helps to read and pass training data gradually during the training
# instead of loading all images in the memory. 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # This loop restarts when one training epoch is complete
		# shuffling the data
        shuffle(samples)
		# reading images and steering angles in batches
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
			# reset the temp data
            images = []
            angles = []
            for batch_sample in batch_samples:
				# read the image address
                name = batch_sample[0]
				# read the corresponding image
                image = cv2.imread(name)
				# resize it and call it center_image
                center_image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
				# preparing a two-channel array (containing binary values for thresholded gradient along Y and S channel)
                filtered_image = np.zeros((80,160,2))
				# applying the sobel transformation and color space filtering and putting the in the channels of the filtered image
                filtered_image[:,:,0] = abs_sobel_thresh(center_image, sobel_kernel=11, thresh=(50, 150)) 
                filtered_image[:,:,1] = hls_select(center_image, thresh=(90, 255))
				# reading the corresponding steering angle from driver behaviour
                center_angle = float(batch_sample[3])
				# appneding the filtered two-channel image and steering angles as well as their flipped versions (for generalization) to the output data
                images.append(filtered_image)
                angles.append(center_angle)
                images.append(np.fliplr(filtered_image))
                angles.append(-center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



# importing other useful packages for training
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Lambda

# From the filtered and resized image (160x80x2), crop some less useful pixels from top an bottom for faster processing
crop_top = 28
crop_bottom = 10
# the format of final image that goes to the first convolution layer 
ch, row, col = 2, 80-crop_top-crop_bottom, 160  # Trimmed image format

# Neural network architecture
model = Sequential()
# cropping unnecessary parts of the image
model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0)), input_shape=(80,160,2)))
# embedding the normalization in the NN: it converts the binary channels from the range [0,1] to the range [-1,1]
model.add(Lambda(lambda x: x*2 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
# convolution layers that gradually become deeper
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), input_shape=(row, col, ch)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2,2)))
model.add(ELU())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
# flattening the outputs and using dense layers up to the final output (i.e. steering angle)
model.add(Flatten())
model.add(Dense(1000))
model.add(ELU())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(20))
model.add(ELU())
# outputting the steering angle
model.add(Dense(1))

# setting optimization metric and optimization method
model.compile(loss='mse', optimizer='adam')
# training the network using the data generators, and training and validation data 
# since I used the flipped versions of each image as well, I choose the sample_per_epcoh = 2*train_samples
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples*2), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

print('training finished')
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# saving the model and weights
model.save_weights('model.h5')
with open('model.json', 'w') as modelfile:
    json.dump(model.to_json(), modelfile)

import gc; gc.collect()




