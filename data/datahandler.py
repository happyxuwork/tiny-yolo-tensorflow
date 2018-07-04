#!/usr/bin/env python3
#import sh
import os
import random
import numpy as np
import cv2

data_path = "./data"

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



def create(input_size=416, flip=1, crop=0.9, angle=10, color = 0.05):
    image_name = random.choice(images_list)
    image_path = os.path.join(images_path, image_name)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join(labels_path, label_name)

    #image
    im = cv2.imread(image_path).astype(np.float)
    h, w, _ = im.shape
        
        #rotate
    rot = random.uniform(-angle, +angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    im = cv2.warpAffine(im, M, (w, h))
    
        #crop
    size = int(min(w, h) * random.uniform(crop, 1))
    x_min = int(random.uniform(0, w - size))
    y_min = int(random.uniform(0, h - size))
    x_max = x_min + size
    y_max = y_min + size
    im = im[y_min:y_max, x_min:x_max, :]

        #flip
    fl = random.random() < 0.5
    if fl:
        im = cv2.flip(im, 1)
    
       #color
    red = random.uniform(1-color, 1+color)
    blu = random.uniform(1-color, 1+color)
    gre = random.uniform(1-color, 1+color)

    col = np.array([blu, gre, red])
    im = im*col
    im[im<0] = 0
    im[im>255] = 255

    image = im
    
    #label

    label = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labeltxt = f.read()
        for objtxt in labeltxt.split("\n"):
            if objtxt == "": continue
            cls, x0, y0, w0, h0, _ = objtxt.split(" ")
            cls = int(cls)
            x0   = float(x0)
            y0   = float(y0)
            w0   = float(w0)
            h0   = float(h0)
            #convert back
            
                #rotate
            rot = np.deg2rad(rot)
            M = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
            x0, y0 = 0.5+np.matmul(M, np.array([x0-0.5, y0-0.5]))
                #w0 h0 remain
            
                #crop
            if x0 < x_min/w or x0 > x_max/w or y0 < y_min/h or y0 > y_max/h: continue
            x0 = (x0*w - x_min)/size
            y0 = (y0*h - y_min)/size
            w0 = w0*w/size
            h0 = h0*h/size

                #flip
            if fl:
                x0 = 1-x0
            
            label.append((cls, x0, y0, w0, h0))
    return image, label

if __name__ == "__main__":
    image, label = create()
    image = image.astype(np.int32)
    for obj in label:
        cls, x0, y0, w0, h0 = obj
        x1 = int((x0 - w0/2)*416)
        x2 = int((x0 + w0/2)*416)
        y1 = int((y0 - h0/2)*416)
        y2 = int((y0 + h0/2)*416)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0),2)

    cv2.imwrite("temp.jpg", image)
    sh.eog("temp.jpg")
    sh.rm("temp.jpg")

def IoU(box1, box2):
    w1, h1 = box1
    w2, h2 = box2
    iou = min(w1, w2) * min(h1, h2)
    return iou

def which_anchor(box):
    anchor = ((10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319))
    dist = []
    for i in range(6):
        dist.append(IoU(anchor[i], box))
    i = dist.index(max(dist))
    return i

def shuffle(batch_size = 1):
    step = 0
    while (1):
        yield step
        
        if (step!=0):
            del Xp
            del Y1p
            del Y2p
        step += 1

        #data augmentation
        image, label = create()
        height, width, depth = image.shape
        classes = 80
        out_height = height//32
        out_width = width//32
        out_depth = 3*(5+classes)
        #anchor = ((10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319))
        Xp = np.memmap("/dev/shm/X_data", dtype = np.float32, mode = "w+", shape = (batch_size, height, width, depth))
        Y1p = np.memmap("/dev/shm/Y1", dtype = np.float32, mode = "w+", shape = (batch_size, out_height, out_width, out_depth))
        Y2p = np.memmap("/dev/shm/Y2", dtype = np.float32, mode = "w+", shape = (batch_size, 2*out_height, 2*out_width, out_depth))
        
        X = image
        Y1 = np.random.random((batch_size, out_height, out_width, out_depth))
        Y2 = np.random.random((batch_size, 2*out_height, 2*out_width, out_depth))
        for i in range(3):
            Y1[:, :, :, i*(out_depth//3)] = 1
            Y2[:, :, :, i*(out_depth//3)] = 1
        #convert label to array
        for obj in label:
            cls, x0, y0, w0, h0 = obj
            box = (w0, h0)
            i = which_anchor(box)
            if (i<3): #anchor1
                x = int(out_width*x0)
                y = int(out_height*y0)
                Y1[0, y, x, 0+i*(out_depth//3)] = 1
                Y1[0, y, x, 1+i*(out_depth//3)] = x0
                Y1[0, y, x, 2+i*(out_depth//3)] = y0
                Y1[0, y, x, 3+i*(out_depth//3)] = w0
                Y1[0, y, x, 4+i*(out_depth//3)] = h0
                Y1[0, y, x, 4:(i+1)*(out_depth//3)] = 0
                Y1[0, y, x, cls] = 1
            else: #anchor2
                i = i - 3
                x = int(2*out_width*x0)
                y = int(2*out_height*y0)
                Y2[0, y, x, 0+i*(2*out_depth//3)] = 1 
                Y2[0, y, x, 1+i*(2*out_depth//3)] = x0
                Y2[0, y, x, 2+i*(2*out_depth//3)] = y0
                Y2[0, y, x, 3+i*(2*out_depth//3)] = w0
                Y2[0, y, x, 4+i*(2*out_depth//3)] = h0
                Y2[0, y, x, 4:(i+1)*(2*out_depth//3)] = 0
                Y2[0, y, x, cls] = 1
        Xp[:] = X[:]
        Y1p[:] = Y1[:]
        Y2p[:] = Y2[:]
