# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:49:34 2018

@author: Bowen
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from functions import load_images, show_image, save_image, batch, predict, InceptionModel

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MomentumIterativeMethod

from keras.applications import vgg16, inception_v3, resnet50, mobilenet

tensorflow_master = ""
checkpoint_path   = "NIPS/inception-v3/inception_v3.ckpt"
input_dir         = "NIPS/images"
max_epsilon       = 16.0
image_width       = 299
image_height      = 299
batch_size        = 1000
eps = 2.0 * max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
num_classes = 1001


categories = pd.read_csv("NIPS/categories.csv")
image_classes = pd.read_csv("NIPS/images.csv")
image_iterator = load_images(input_dir, batch_shape)

# get first batch of images
filenames, images = next(image_iterator)

image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
true_classes = image_metadata["TrueLabel"].tolist()
target_classes = true_labels = image_metadata["TargetClass"].tolist()

#true_classes_names = (pd.DataFrame({"CategoryId": true_classes})
#                        .merge(categories, on="CategoryId")["CategoryName"].tolist())

true_classes_names = []
for i in true_classes:
    true_classes_names.append(categories['CategoryName'][i-1])


#target_classes_names = (pd.DataFrame({"CategoryId": target_classes})
#                          .merge(categories, on="CategoryId")["CategoryName"].tolist())

target_classes_names = []
for i in target_classes:
    target_classes_names.append(categories['CategoryName'][i-1])


print("Here's an example of one of the images in the development set")
show_image(images[0])

print(true_classes_names[0].split(',')[0].replace(' ', '_')) 
#print(categories['CategoryName'][0])

# Saves the image in order for analysis and spit out their correct category
for i in range(len(images)):
    save_image(images[i], i, 'C:/Users/Bowen/Desktop/Project/image-perturbation-defense/NIPS/test/')
    correct_label = true_classes_names[i].split(',')[0].replace(' ', '_')  # Convert ['giant panda, panda etc....'] to 'giant_panda'
    with open('true_classes.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([correct_label])




slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
batch_shape = [32, image_height, image_width, 3]
image_iterator = load_images(input_dir, batch_shape)

# Purturbs the image in a batches of 32 to avoid memory issue
for x in batch(range(0, 1000), 32):
    print(x)

    filenames, images = next(image_iterator)
    
    all_images_target_class = {image_metadata["ImageId"][i]+".png": image_metadata["TargetClass"][i] for i in image_metadata.index}

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        model = InceptionModel(num_classes)

        carlini = CarliniWagnerL2(model)
        jacobian = SaliencyMapMethod(model)
        fgsm = FastGradientMethod(model)
        momentum = MomentumIterativeMethod(model)
        x_adv = momentum.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
                          scaffold=tf.train.Scaffold(saver=saver),
                          checkpoint_filename_with_path=checkpoint_path,
                          master=tensorflow_master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            nontargeted_images = sess.run(x_adv, feed_dict={x_input: images})

    print("The original image is on the left, and the nontargeted adversarial image is on the right. They look very similar, don't they? It's very clear both are gondolas")
    show_image(np.concatenate([images[1], nontargeted_images[1]], axis=1))

    for i in range(len(nontargeted_images)):
        save_image(nontargeted_images[i], x[i], 'NIPS/perturbed2/')




vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

for i in range(1000):
    predict(inception_v3, inception_model, 'NIPS/perturbed2/', 1, i)
