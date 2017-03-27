# importing the necessary packages
import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import tensorflow as tf
tf.python.control_flow_ops = tf
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


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

##############################################################################################



# the function for processing simulation frames and applying the corresponding steering angle via the trained NN
@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    #converting the image to array for preprocessing
    image_array = np.asarray(image)
    # resizing the image from 320x160 to 160x80
    image_array_resized = cv2.resize(image_array,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    # preparing a two-channel array (containing binary values for thresholded gradient along Y and S channel)
    filtered_image = np.zeros((80,160,2))
	# applying the sobel transformation and color space filtering and putting the in the channels of the filtered image
    filtered_image[:,:,0] = abs_sobel_thresh(image_array_resized, sobel_kernel=11, thresh=(50, 150)) 
    filtered_image[:,:,1] = hls_select(image_array_resized, thresh=(90, 255))
    transformed_image_array = filtered_image[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. 
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. 
    throttle = 0.3
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        #model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)