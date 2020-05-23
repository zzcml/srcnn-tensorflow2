from srcnn import SRCNN
from utils import (
  read_data,
  input_setup,
  imsave,
  merge
)
import numpy as np
import tensorflow as tf

import pprint
import os
import argparse
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

parser = argparse.ArgumentParser(description='SRCNN Training')
parser.add_argument("--epoch", default=15000, type=int, help="Number of epoch [15000]")
parser.add_argument("--batch_size", default=128, type=int, help="The size of batch images [128]")
parser.add_argument("--image_size", default=33, type=int, help="The size of image to use [33]")
parser.add_argument("--label_size", default=21, type=int, help="The size of label to produce [21]")
parser.add_argument("--learning_rate", default=1e-4, type=int, help="The learning rate of gradient descent algorithm [1e-4]")
parser.add_argument("--c_dim", default=1, type=int, help="Dimension of image color. [1]")
parser.add_argument("--scale", default=3, type=int, help="The size of scale factor for preprocessing input image [3]")
parser.add_argument("--stride", default=14, type=int, help="The size of stride to apply input image [14]")
parser.add_argument("--checkpoint_dir", default="checkpoint", type=str, help="Name of checkpoint directory [checkpoint]")
parser.add_argument("--sample_dir", default="sample", type=str, help="Name of sample directory [sample]")
# flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
args, unknown = parser.parse_known_args()

pp = pprint.PrettyPrinter()

def createmodel(args):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (9, 9), kernel_initializer='normal', strides=1, padding='VALID',
                                    activation='relu', input_shape=[args.image_size, args.image_size, args.c_dim], name='conv1'))
    model.add(tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer='normal', strides=1, padding='VALID',
                                    activation='relu', name='conv2'))
    model.add(tf.keras.layers.Conv2D(1, (5, 5), kernel_initializer='normal', strides=1, padding='VALID',
                                    activation='relu', name='conv3'))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=args.learning_rate),
                      loss=tf.losses.MSE)
    return model

pp.pprint(args)
# if not os.path.exists(FLAGS.checkpoint_dir):
#     os.makedirs(FLAGS.checkpoint_dir)
# if not os.path.exists(FLAGS.sample_dir):
#     os.makedirs(FLAGS.sample_dir)
# if FLAGS.is_train:
data_dir = 'checkpoint/train.h5'
# else:
#   data_dir = os.path.join('./{}'.format(FLAGS.checkpoint_dir), "test.h5")
train_data, train_label = read_data(data_dir)
srcnn = createmodel(args)
logging = TensorBoard(log_dir=args.checkpoint_dir)
checkpoint = ModelCheckpoint(os.path.join(args.checkpoint_dir + "%s_%s" % ("srcnn", args.label_size), 'ep{epoch:03d}-loss{loss:.3f}.h5'),
                                 monitor='loss', save_weights_only=True, save_best_only=True, period=500)
srcnn.fit(train_data, train_label, epochs=args.epoch, callbacks=[logging, checkpoint])
