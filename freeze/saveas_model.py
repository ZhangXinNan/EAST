#encoding=utf8
import cv2
import time
import math
import os
import glob
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import locality_aware_nms as nms_locality
import lanms


# tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_model_dir', '../models/east_icdar2015_resnet_v1_50_rbox_ckpt', '')
tf.app.flags.DEFINE_string('model_name', '/model.ckpt', '')


import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_model_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            saver.save(sess, os.path.join(FLAGS.output_model_dir, FLAGS.model_name))

if __name__ == '__main__':
    tf.app.run()
