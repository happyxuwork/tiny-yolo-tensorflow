#!/usr/bin/env python3
import tensorflow as tf

g1 = tf.Graph()

with g1.as_default() as g:
    with g.name_scope("YOLO"):
        def conv(n, in_name, out_channels, kernel_size, stride, padding="SAME", nonlin="relu", binary=0):
            in_tensor = g.get_tensor_by_name(in_name)
            batch, height, width, in_channels = in_tensor.get_shape().as_list()
            with g.name_scope("conv_{}".format(n)):
                kernel = tf.Variable(tf.random_normal(shape = [kernel_size, kernel_size, in_channels, out_channels]), dtype = tf.float32, name = "kernel")
                scale = tf.Variable(tf.random_normal(shape = [1,]), dtype = tf.float32, name = "scale")
                bias = tf.Variable(tf.random_normal(shape = [1,]), dtype = tf.float32, name = "bias")
                '''
                conv
                batchnorm + bias + scale
                nonlin
                '''
                strides = (1, stride, stride, 1)
                conv = tf.nn.conv2d(in_tensor, kernel, strides, padding, name = "conv")
                mean_conv, var_conv = tf.nn.moments(conv, axes = [1,2,3], keep_dims = True)
                batchnorm = tf.nn.batch_normalization(conv, mean_conv, var_conv, bias, scale, 1e-100)
                if nonlin == "relu":
                    nonlin = tf.nn.leaky_relu(batchnorm)
                elif nonlin == "sigmoid":
                    nonlin = tf.sigmoid(batchnorm)
                elif nonlin == "linear":
                    nonlin = tf.identity(batchnorm)
                else:
                    raise Exception(" \"{}\" is not a nonlinear function!".format(nonlin))
                conv = tf.identity(nonlin, name = "out")
            return conv
 
        def maxpool(n, in_name, kernel_size, stride, padding="SAME"):
            in_tensor = g.get_tensor_by_name(in_name)
            batch, height, width, in_channels = in_tensor.get_shape().as_list()
            with g.name_scope("maxpool_{}".format(n)):
                ksize = [1, kernel_size, kernel_size, 1]
                strides = [1, stride, stride, 1]
                '''
                maxpool
                '''
                maxpool = tf.nn.max_pool(in_tensor, ksize, strides, padding)
                maxpool = tf.identity(maxpool, name = "out")
            return maxpool
 
        def route(n, n1_name, n2_name):
 
            if (n2_name==None):
                n1 = g.get_tensor_by_name(n1_name)
                route = tf.identity(n1)
            else:
                n1 = g.get_tensor_by_name(n1_name)
                n2 = g.get_tensor_by_name(n2_name)
                route = tf.concat([n1, n2], 3)
            with g.name_scope("route_{}".format(n)):
                route = tf.identity(route, name = "out")
            return route
 
        def upsample(n, in_name, stride):
            in_tensor = g.get_tensor_by_name(in_name)
            batch, height, width, in_channels = in_tensor.get_shape().as_list()
            out_channels = in_channels
            with g.name_scope("upsample_{}".format(n)):
                kernel = tf.ones([stride, stride, in_channels, out_channels], name = "kernel")
                output_shape = [batch, stride*height, stride*width, in_channels]
                strides = [1, stride, stride, 1]
                padding = "SAME"
                unsample = tf.nn.conv2d_transpose(in_tensor, kernel, output_shape, strides, name = "out")
            return unsample
 
        def yolo(n, in_name, anchor, thresh=0.5):#in tensor has shape (batch, height, width, 255)
            in_tensor = g.get_tensor_by_name(in_name)
            batch, height, width, in_channels = in_tensor.get_shape().as_list()
            split = tf.split(in_tensor, 3, axis = 3)
            new_split = []
            for i in range(3):
                xy = split[i][:, :, :, 0:2]
                xy = tf.sigmoid(xy)
                wh = split[i][:, :, :, 2:4]
                wh = tf.constant(anchor[i], dtype = tf.float32) * tf.exp(wh)
                onc = split[i][:, :, :, 4: ]
                onc = tf.sigmoid(onc)
                new_split.append(xy)
                new_split.append(wh)
                new_split.append(onc)
                #x,y,w,h,obj,classes
            
            with g.name_scope("yolo_{}".format(n)):
                yolo = tf.concat(split, 3, name = "out")
            return yolo

        height = 416
        width = 416
        anchor1 = [(81,82),  (135,169),  (344,319)]
        anchor2 = [(10,14),  (23,27),  (37,58)]
        classes = 80

        out_height = height//32
        out_width = width//32
        out_depth = 3*(5 + classes)

        X = tf.placeholder(shape = (1, height, width, 3), dtype = tf.float32, name = "X")
        Y1 = tf.placeholder(shape = (1, out_height, out_width, out_depth), dtype = tf.float32, name = "Y1")
        Y2 = tf.placeholder(shape = (1, 2*out_height, 2*out_width, out_depth), dtype = tf.float32, name = "Y2")
        #0
        conv_0 = conv(0, "YOLO/X:0", 16, 3, 1)
        #1
        maxpool(1, "YOLO/conv_0/out:0", 2, 2)
        #2
        conv(2, "YOLO/maxpool_1/out:0", 32, 3, 1)
        #3
        maxpool(3, "YOLO/conv_2/out:0", 2, 2)
        #4
        conv(4, "YOLO/maxpool_3/out:0", 64, 3, 1)
        #5
        maxpool(5, "YOLO/conv_4/out:0", 2, 2)
        #6
        conv(6, "YOLO/maxpool_5/out:0", 128, 3, 1)
        #7
        maxpool(7, "YOLO/conv_6/out:0", 2, 2)
        #8
        conv(8, "YOLO/maxpool_7/out:0", 256, 3, 1)
        #9
        maxpool(9, "YOLO/conv_8/out:0", 2, 2)
        #10
        conv(10, "YOLO/maxpool_9/out:0", 512, 3, 1)
        #11
        maxpool(11, "YOLO/conv_10/out:0", 2, 1)
        #12
        conv(12, "YOLO/maxpool_11/out:0", 1024, 3, 1)
        #13
        conv(13, "YOLO/conv_12/out:0", 256, 1, 1)
        #14
        conv(14, "YOLO/conv_13/out:0", 512, 3, 1)
        #15
        conv(15, "YOLO/conv_14/out:0", 255, 1, 1)
        #16
        yolo(16, "YOLO/conv_15/out:0", anchor1)
        #17
        route(17, "YOLO/conv_13/out:0", None)
        #18
        conv(18, "YOLO/route_17/out:0", 128, 1, 1)
        #19
        upsample(19, "YOLO/conv_18/out:0", 2)
        #20
        route(20, "YOLO/upsample_19/out:0", "YOLO/conv_8/out:0")
        #21
        conv(21, "YOLO/route_20/out:0", 256, 3, 1)
        #22
        conv(22, "YOLO/conv_21/out:0", 255, 1, 1)
        #23
        yolo(23, "YOLO/conv_22/out:0", anchor2)

import os
import shutil
if os.path.exists("./graph"): shutil.rmtree("./graph")
os.mkdir("./graph")

tf.summary.FileWriter("./graph", g)
saver = tf.train.Saver()
with tf.Session(graph = g) as sess:
    saver.save(sess, "./graph/tiny-yolo.ckpt")


