#!/usr/bin/env python3
import tensorflow as tf
g1 = tf.Graph()

with g1.as_default() as g:

    def conv(n, in_name, out_channels, kernel_size, stride, padding="SAME", nonlin="relu", binary=0):
        in_tensor = g.get_tensor_by_name(in_name)
        batch, height, width, in_channels = in_tensor.get_shape().as_list()
        with g.name_scope("conv{}".format(n)):
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
            mean_conv, var_conv = tf.nn.moments(conv, axes = [2,3,4], keep_dims = True)
            batchnorm = tf.nn.batch_normalization(conv, mean_conv, var_conv, bias, scale, 1e-100)
            if nonlin == "relu":
                nonlin = tf.nn.leaky_relu(batchnorm, name = "out")
            elif nonlin == "sigmoid":
                nonlin = tf.sigmoid(batchnorm, name = "out")
            elif nonlin == "linear":
                nonlin = tf.identity(batchnorm, name = "out")
            else:
                raise Exception(" \"{}\" is not a nonlinear function!".format(nonlin))
        return nonlin

    def max(n, in_name, kernel_size, stride, padding="VALID"):
        in_tensor = g.get_tensor_by_name(in_name)
        batch, height, width, in_channels = in_tensor.get_shape().as_list()
        with g.name_scope("max_{}".format(n)):
            ksize = [1, kernel_size, kernel_size, 1]
            strides = [1, stride, stride, 1]
            '''
            max
            '''
            max_pool = tf.nn.max_pool(in_tensor, ksize, strides, padding, name = "out")
        return max_pool

    def route(n, n1_name, n2_name):

        if (n2_name==None):
            n1 = g.get_tensor_by_name(n1_name)
            with g.name_scope("route_{}".format(n)):
                route = tf.identity(n1, name = "out")
        else:
            n1 = g.get_tensor_by_name(n1_name)
            n2 = g.get_tensor_by_name(n2_name)
            with g.name_scope("route_{}".format(n)):
                route = tf.concat([n1, n2], 3, name = "out")
        return route

    def upsample(n, in_name, stride):
        in_tensor = g.get_tensor_by_name(in_name)
        batch, height, width, in_channels = in_tensor.get_shape().as_list()
        with g.name_scope("unsample_{}".format(n)):
            kernel = tf.ones([stride, stride, 1, 1], name = "kernel")
            output_shape = [batch, stride*height, stride*width, in_channles]
            strides = [1, stride, stride, 1]
            padding = "SAME"
            unsample = tf.nn.conv2d_transpose(in_tensor, kernel, output_shape, strides, name = "out")
        return unsample

    def yolo()

