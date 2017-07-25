#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pprint

import tensorflow as tf

from util import read_data
from model import RescorlaWagner


flags = tf.app.flags

flags.DEFINE_string('data_file', None, 'file with input data')
flags.DEFINE_boolean('add_background', False, 'add background cue')
flags.DEFINE_float('alpha_beta', 0.1, 'alpha * beta')
flags.DEFINE_float('n_steps', 1e5, 'number of steps to compute')
flags.DEFINE_string('from_checkpoint', None, 'checkpoint to start with')
flags.DEFINE_float('n_steps_checkpoint', 1e5, 'checkpoint to start with')
flags.DEFINE_string('name', 'rw', 'model_name')

FLAGS = flags.FLAGS


def main(_):
    pprint.pprint(FLAGS.__flags)
    cues, cues_dict, outcomes, outcomes_dict, probs = read_data(
        FLAGS.data_file, FLAGS.add_background)
    model = RescorlaWagner(cues_dict, outcomes_dict,
                           alpha_beta=FLAGS.alpha_beta,
                           name=FLAGS.name)

    if FLAGS.from_checkpoint is None:
        model.init_model()
    else:
        model.init_model(FLAGS.from_checkpoint)

    model.train(int(FLAGS.n_steps), cues, outcomes, probs,
                int(FLAGS.n_steps_checkpoint))

if __name__ == '__main__':
    tf.app.run()
