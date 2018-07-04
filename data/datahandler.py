#!/usr/bin/env python3
#import sh
import os
import random
import numpy as np
import cv2

data_path = "../data"

images_path = os.path.join(data_path, "images")
labels_path = os.path.join(data_path, "labels")
'''
count = 0
import pdb
for image_name in os.listdir(images_path):

    image_path = os.path.join(images_path, image_name)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join(labels_path, label_name)

    if not os.path.exists(label_path):
        count += 1
        print(count, image_path)
        #sh.touch(label_path)
'''
images_list = os.listdir(images_path)

def create(input_size=416, flip=1, crop=0.9, angle=10, color = 0.10):
    image_name = random.choice(images_list)
    image_path = os.path.join(images_path, image_name)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join(labels_path, label_name)

    #image
    im = cv2.imread(image_path).astype(np.float)
    h, w, _ = im.shape
    '''
        #rotate
    rot = random.uniform(-angle, +angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    im = cv2.warpAffine(im, M, (w, h))
    '''
        #crop
    size = int(min(w, h) * random.uniform(crop, 1))
    x_min = int(random.uniform(0, w - size))
    y_min = int(random.uniform(0, h - size))
    x_max = x_min + size
    y_max = y_min + size
    im = im[y_min:y_max, x_min:x_max, :]

        #flip
    fl = random.random() < 0.5:
    if fl:
        im = cv2.flip(im, 1)
    
       #color
    red = random.uniform(1-color, 1+color)
    blu = random.uniform(1-color, 1+color)
    gre = random.uniform(1-color, 1+color)

    col = np.array([blu, gre, red])
    im = im*col

    image = im
    
    #label
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labeltxt = f.read()



def generator(batch_size = 1):
    while (1)
        yield 1

        #data augmentation
