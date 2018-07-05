#!/usr/bin/env python3
from data.datahandler import shuffle
import tensorflow as tf
import numpy as np
import time
import os
import sys
import shutil

def read_batch():
    batch_size = 1
    height = 416
    width = 416
    depth = 3
    classes = 80
    out_height = height//32
    out_width = width//32
    out_depth = 3*(5+classes)

    Xp = np.memmap("/dev/shm/X", dtype = np.float32, mode = "r", shape = (batch_size, height, width, depth))
    Y1p = np.memmap("/dev/shm/Y1", dtype = np.float32, mode = "r", shape = (batch_size, out_height, out_width, out_depth))
    Y2p = np.memmap("/dev/shm/Y2", dtype = np.float32, mode = "r", shape = (batch_size, 2*out_height, 2*out_width, out_depth))
    return Xp, Y1p, Y2p



saver = tf.train.import_meta_graph("./graph/tiny-yolo.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "./graph/tiny-yolo.ckpt")
    g = sess.graph
    with g.name_scope("TRAINER"):
        X = g.get_tensor_by_name("YOLO/input:0")
        batch_size, height, width, in_channels = X.get_shape().as_list()
        classes = 80
        out_height = height//32
        out_width = width//32
        out_channels = 3*(5+classes)
        h1 = g.get_tensor_by_name("YOLO/output1:0")
        h2 = g.get_tensor_by_name("YOLO/output2:0")
        Y1 = tf.placeholder(shape = (batch_size, out_height, out_width, out_channels), dtype = tf.float32, name = "groundtruth1")
        Y2 = tf.placeholder(shape = (batch_size, 2*out_height, 2*out_width, out_channels), dtype = tf.float32, name = "groundtruth2")
        #loss1
        Lcoord = 1

        split10 = tf.split((h1-Y1)**2,3, axis = 3)
        split05 = tf.split((h1**0.5-Y1**0.5)**2, 3, axis = 3)
        Y1split = tf.split(Y1, 3, axis = 3)
    
        for i in range(3):
            p_loss = tf.reduce_mean(split10[i][:,:,:,0]) #confidence
            class_loss = tf.reduce_mean(split10[i][:,:,:,5:])
            xy_loss = tf.reduce_mean(split10[i][:,:,:,1:3] * tf.tile(tf.expand_dims(Y1split[i][:,:,:,0],3), [1,1,1,2]))
            wh_loss = tf.reduce_mean(split05[i][:,:,:,3:5] * tf.tile(tf.expand_dims(Y1split[i][:,:,:,0],3), [1,1,1,2]))
            loss = (p_loss + class_loss) + Lcoord*(xy_loss+wh_loss)
            loss = tf.identity(loss, name = "loss")
            
        optimizer = tf.train.AdamOptimizer(learning_rate = 1e-5)
        trainer = optimizer.minimize(loss, name = "trainer")

    if os.path.exists("./train_graph"):
        inp = input("Are you sure to delete the old graph? [Y, N] ")
        if inp.lower() == "y":
            shutil.rmtree("./train_graph")
        else:
            sys.exit(0)
    os.mkdir("./train_graph")

    train_writer = tf.summary.FileWriter("./train_graph", g)
    saver = tf.train.Saver()
    tf.summary.histogram("loss", loss)
    merge = tf.summary.merge_all()

    hm_steps = 400000
    sess.run(tf.global_variables_initializer())





    for batch in shuffle():
        step, Xp, Y1p, Y2p = batch
        if step == 0:
            time.sleep(1)
            continue
        debugger = tf.is_nan(loss)
        while (1):
            if sess.run(debugger, feed_dict = {X:Xp, Y1:Y1p, Y2:Y2p}):
                break
            else:
                print("Re-random variables!")
                sess.run(tf.global_variables_initializer())



        _ , lossp, summary = sess.run([trainer, loss, merge], feed_dict = {X: Xp, Y1: Y1p, Y2:Y2p})

        train_writer.add_summary(summary, step)
        print("Step {} : loss {}".format(step, lossp))

        if (step % 80000 ==0):
            saver.save(sess, "./train_graph/tiny-yolo-{}.ckpt".format(step))
        



