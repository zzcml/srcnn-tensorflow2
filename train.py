
from utils import (
  read_data,
  input_setup,
  imsave,
  merge,
  get_last_weights
)
import numpy as np
import datetime
import tensorflow as tf
import time
import pprint
import os
import argparse
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='SRCNN Training')
parser.add_argument("--epoch", default=150, type=int, help="Number of epoch [15000]")
parser.add_argument("--batch_size", default=16, type=int, help="The size of batch images [128]")
parser.add_argument("--image_size", default=33, type=int, help="The size of image to use [33]")
parser.add_argument("--label_size", default=21, type=int, help="The size of label to produce [21]")
parser.add_argument("--learning_rate", default=1e-4, type=int,
                    help="The learning rate of gradient descent algorithm [1e-4]")
parser.add_argument("--c_dim", default=1, type=int, help="Dimension of image color. [1]")
parser.add_argument("--scale", default=3, type=int, help="The size of scale factor for preprocessing input image [3]")
parser.add_argument("--stride", default=14, type=int, help="The size of stride to apply input image [14]")
parser.add_argument("--checkpoint_dir", default="checkpoint/", type=str, help="Name of checkpoint directory [checkpoint]")
parser.add_argument("--sample_dir", default="sample", type=str, help="Name of sample directory [sample]")
parser.add_argument("-w", "--load_weights", default='last', type=str, help="whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint")
parser.add_argument("--save_path", default='checkpoint/models/', type=str)
parser.add_argument("--is_train", default=True, type=bool, help="True for training, False for testing [True]")
# parser.add_argument("--is_train", default=False, type=bool, help="True for training, False for testing [True]")
args, unknown = parser.parse_known_args()

pp = pprint.PrettyPrinter()
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

def createmodel(args):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (9, 9), kernel_initializer='normal', strides=1, padding='VALID',
                                    activation='relu', input_shape=[args.image_size, args.image_size, args.c_dim],
                                    name='conv1'))
    model.add(tf.keras.layers.Conv2D(32, (1, 1), kernel_initializer='normal', strides=1, padding='VALID',
                                    activation='relu', name='conv2'))
    model.add(tf.keras.layers.Conv2D(1, (5, 5), kernel_initializer='normal', strides=1, padding='VALID',
                                    activation='relu', name='conv3'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate),
                      loss=tf.losses.MSE)
    return model

pp.pprint(args)
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.sample_dir, exist_ok=True)
if args.is_train:
    input_setup(args)
    data_dir = 'checkpoint/train.h5'
    train_data, train_label = read_data(data_dir)
    srcnn = createmodel(args)
    # load last weights
    if args.load_weights is not None:
        if args.load_weights.endswith('.h5'):
            weights_path = args.load_weights
        else:
            weights_path = get_last_weights(args.save_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = srcnn.load_weights(weights_path)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
    current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    model_path = 'SRCNN.h5'
    saved_model = tf.keras.callbacks.ModelCheckpoint(args.save_path + 'ep_{epoch:03d}.h5', monitor='loss',
                                                     save_weights_only=True, save_best_only=True, period=5)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    start_time = time.time()
    history = srcnn.fit(train_data, train_label, batch_size=args.batch_size, validation_split=0.2,
                        epochs=args.epoch, initial_epoch=last_step, callbacks=[saved_model, tensorboard], verbose=2)
    print('spending time:' + str(time.time() - start_time))
    # plot_graphs(history, "val_loss")
    plot_graphs(history, "loss")
else:
    nx, ny = input_setup(args)
    data_dir = 'checkpoint/test.h5'
    weights_path = 'checkpoint/ep150-loss0.005.h5'
    test_data, test_label = read_data(data_dir)
    print(test_data.shape)
    srcnn = createmodel(args)
    srcnn.load_weights(weights_path)
    result = srcnn.predict(test_data)
    print(result.shape)
    # result = srcnn.evaluate(test_data, test_label)

    result = merge(result, [nx, ny])
    print(result.shape)
    image_path = os.path.join(os.getcwd(), args.sample_dir)
    image_path = os.path.join(image_path, "test_image.png")
    imsave(result, image_path)
