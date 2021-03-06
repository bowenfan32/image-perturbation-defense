# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:37:41 2019

@author: Bowen
"""


import os
from io import BytesIO
import IPython.display
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave

from tensorflow.contrib.slim.nets import inception


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape) # initialize a blank image with given shape of [16, 299, 299, 3]
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float)*2.0/255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def show_image(a, fmt='png'):
    a = np.uint8((a+1.0)/2.0*255.0)
    f = BytesIO()
    Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
    
def save_image(a, num, path, fmt='png'):
    a = np.uint8((a+1.0)/2.0*255.0)
    Image.fromarray(a).save(path + str(num) + '.png', fmt)
    print(str(num))
   
slim = tf.contrib.slim
class InceptionModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                            x_input, num_classes=self.num_classes, is_training=False,
                            reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        return probs
    
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
        
        
        
import cv2
import csv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt


def to_csv(result):
    with open('result.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(result)

def predict(model_name, model, path, compression_percent, image_num):
    
    filename = path + str(image_num)+'.png'
    # load an image in PIL format
    original = load_img(filename, target_size=(224,224))
#    print('PIL image size',original.size)
#    plt.imshow(original)
#    plt.show()
    

    data = cv2.imread( path + str(image_num)+'.png')
    compressed_dimension = 224 * compression_percent
    small_img = cv2.resize(data, dsize=(int(compressed_dimension), int(compressed_dimension)), interpolation=cv2.INTER_AREA)
    big_img = cv2.resize(small_img, dsize=(224,224), interpolation=cv2.INTER_LANCZOS4)
#    RGB_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    #plt.imshow(RGB_img)
    #plt.show()
     
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(small_img)
    plt.imshow(np.uint8(numpy_image))
    #plt.show()
    #print('numpy array size',numpy_image.shape)
    
    numpy_image = img_to_array(big_img)
     
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
#    print('image batch size', image_batch.shape)
    #plt.imshow(np.uint8(image_batch[0]))
    #plt.show()
    
    # prepare the image for the VGG model
    processed_image = model_name.preprocess_input(image_batch.copy())
     
    # get the predicted probabilities for each class
    predictions = model.predict(processed_image)
    
    # print predictions
    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions)
    
    result = (label[0][0][1], str(round(label[0][0][2]*100, 2))+'%')
    
    to_csv(result)
#    print (result)
    
    
#    print("UNMODIFIED IMAGE (left)", 
#          "\n\tPredicted class:", label[0][0][1], 
#          "\n\tTrue class:     ", true_classes_names[i]) 

    