# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:49:34 2018

@author: Bowen
"""

import os
from cleverhans.attacks import FastGradientMethod
from io import BytesIO
import IPython.display
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim
tensorflow_master = ""
checkpoint_path   = "NIPS/inception-v3/inception_v3.ckpt"
input_dir         = "NIPS/images/"
max_epsilon       = 16.0
image_width       = 299
image_height      = 299
batch_size        = 1000

eps = 2.0 * max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
num_classes = 1001

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
    
    
def save_image(a, num, fmt='png'):
    a = np.uint8((a+1.0)/2.0*255.0)
    Image.fromarray(a).save('C:/Users/Bowen/Desktop/Project/image-perturbation-defense/NIPS/perturbed/' + str(i) + '.png', fmt)
   

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
    
    
    
    
categories = pd.read_csv("NIPS/categories.csv")
image_classes = pd.read_csv("NIPS/images.csv")
image_iterator = load_images(input_dir, batch_shape)

# get first batch of images
filenames, images = next(image_iterator)

image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes,
                                                                              on="ImageId")
true_classes = image_metadata["TrueLabel"].tolist()
target_classes = true_labels = image_metadata["TargetClass"].tolist()
true_classes_names = (pd.DataFrame({"CategoryId": true_classes})
                        .merge(categories, on="CategoryId")["CategoryName"].tolist())
target_classes_names = (pd.DataFrame({"CategoryId": target_classes})
                          .merge(categories, on="CategoryId")["CategoryName"].tolist())

print("Here's an example of one of the images in the development set")
show_image(images[0])
print(target_classes_names[0])
print(true_classes_names[0].split(',')[0].replace(' ', '_')) # Convert ['giant panda, panda etc....'] to 'giant_panda'

for i in range(len(images)):
    save_image(images[i], i)
    


# Perturbs the image various methods
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    model = InceptionModel(num_classes)

    carlini = CarliniWagnerL2(model)
    jacobian = SaliencyMapMethod(model)
    fgsm  = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, clip_min=-1., clip_max=1.)

    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
                      scaffold=tf.train.Scaffold(saver=saver),
                      checkpoint_filename_with_path=checkpoint_path,
                      master=tensorflow_master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
        nontargeted_images = sess.run(x_adv, feed_dict={x_input: images})

print("The original image is on the left, and the nontargeted adversarial image is on the right. They look very similar, don't they? It's very clear both are gondolas")


import keras
import numpy as np
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

def to_csv(result):
    import csv
    with open('result.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(result)

def predict(model_name, model, compression_percent, image_num):
    
    filename = 'NIPS/perturbed/'+ str(image_num)+'.png'
    # load an image in PIL format
    original = load_img(filename, target_size=(224,224))
    print('PIL image size',original.size)
#    plt.imshow(original)
#    plt.show()
    

    data = cv2.imread('NIPS/perturbed/'+ str(image_num)+'.png')
    compressed_dimension = 224 * compression_percent
    small_img = cv2.resize(data, dsize=(int(compressed_dimension), int(compressed_dimension)), interpolation=cv2.INTER_AREA)
    big_img = cv2.resize(small_img, dsize=(224,224), interpolation=cv2.INTER_LANCZOS4)
    RGB_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
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
    print('image batch size', image_batch.shape)
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
    print (result)
    
    
for i in range(1000):
    
    predict(inception_v3, inception_model, 0.6 ,i)

