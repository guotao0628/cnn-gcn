'''
Created on 2018-9-11

@author: 74510
'''
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import get_batch
from alexnet import *
from openpyxl import workbook
import time


def test(test_file):
    log_dir = 'log/'
#     image_arr = get_one_image(test_file)
    image=Image.open(test_file).convert("RGB")
#     image=tf.cast(image,tf.string)
    max_index=''
    with tf.Graph().as_default():
#         image = tf.cast(image_arr, tf.float32)
#         image=tf.image.resize_image_with_crop_or_pad(image,224,224)
#         image=tf.image.decode_jpeg(image,channels=3)
        image=tf.image.resize_images(image,[224,224])
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,224, 224, 3])
        print(image.shape)
        conv1,conv2,conv3,conv4,conv5,pool1,pool2,pool3= inference(image,1,5,train=False)
#         features=inference(image,1,5,train=False)
#         logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32,shape = [224,224,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint')
#             prediction=sess.run(logits)
            feature_conv1=sess.run(conv1)
            feature_conv2=sess.run(conv2)
            feature_conv3=sess.run(conv3)
            feature_conv4=sess.run(conv4)
            feature_conv5=sess.run(conv5)
            feature_pool1=sess.run(pool1)
            feature_pool2=sess.run(pool2)
            feature_pool3=sess.run(pool3)
           
           
    return feature_conv1,feature_conv2,feature_conv3,feature_conv4,feature_conv5,feature_pool1,feature_pool2,feature_pool3
# converimage(feature,image_dir,j)
def converimage(feature,image_dir,j):
    print(type(feature))
    feature1=0
    dim=feature.shape[3]
    print(dim)
    for i in range(dim):
        feature1+=feature[0][:,:,i]
        print(feature1.shape)
#     vis_square(feature1, padsize=1, padval=0)
    image = Image.fromarray(feature1)
#     image = image.convert('RGB')
#     image.save(image_dir+'\{}.jpg'.format(j))
    plt.imshow(image,cmap=plt.cm.jet)
    plt.imsave(image_dir+'\{}.png'.format(j),image,cmap=plt.cm.jet)
#     plt.savefig()
    plt.pause(1)
    plt.close()

image_dir=r'F:\peach\images'

conv1,conv2,conv3,conv4,conv5,pool1,pool2,pool3=test(os.path.join(image_dir,'IMG_5794.JPG'))
converimage(conv3,image_dir,3)
