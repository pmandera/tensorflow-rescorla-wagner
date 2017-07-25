#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from util import reverse_dictionary


def get_events(cues, outcomes, probs, size):
    """Select random events from the events pool."""
    idx = np.random.choice(len(probs), p=probs, size=size)
    return (cues[idx], outcomes[idx])


class RescorlaWagner(object):
    def __init__(self, cues_dict, outcomes_dict, alpha_beta=0.1, name='rw',
                 session=tf.Session()):
        self.cues_rev_dict = reverse_dictionary(cues_dict)
        self.outcomes_rev_dict = reverse_dictionary(outcomes_dict)
        self.cues_num = max(cues_dict.values()) + 1
        self.outcomes_num = max(outcomes_dict.values()) + 1
        self.name = name

        self.alpha_beta = alpha_beta

        self.summaries_dir = os.path.join('summaries/', name)

        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)

        self.session = session

        self.build_model()
        self.build_summaries()

    def build_model(self):
        """Build the model."""
        with tf.name_scope('rw_model'):
            self.train_inputs = tf.placeholder(tf.int32)
            self.train_outputs = tf.placeholder(tf.int32)

            # cue-outcome weights
            self.embeddings = tf.Variable(tf.random_uniform(
                [self.cues_num, self.outcomes_num], -0.001, 0.001))

            # only cues that are observed will be observed
            # we can treat them as embeddings
            self.embed = tf.nn.embedding_lookup(self.embeddings,
                                                self.train_inputs)

            self.outputs = tf.reduce_sum(self.embed, 0)
            self.targets = tf.reduce_sum(
                tf.one_hot(self.train_outputs, self.outcomes_num), 0)

            # minimize difference beetween predicted and observed outcomes
            self.loss = tf.reduce_mean(
                tf.square(tf.subtract(self.targets, self.outputs)))

            # Rescorla-Wagner update rule is equivalent to gradient descent
            self.optimizer = tf.train.GradientDescentOptimizer(
                self.alpha_beta).minimize(self.loss)

    def build_summaries(self):
        """Build summaries for monitoring the training progress."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('weights_histogram', self.embed)

            self.merged_summaries = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.summaries_dir,
                                                self.session.graph)

    def init_model(self, checkpoint=None):
        """Load existing model or initialize."""
        if checkpoint:
            self.load(checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

    def train_trial(self, cues, outcomes):
        """Run one training trial."""

        _, loss_val, summaries = self.session.run(
            [self.optimizer, self.loss, self.merged_summaries],
            {self.train_inputs: cues, self.train_outputs: outcomes})

        return loss_val, summaries

    def train(self, n_steps, cues, outcomes, probs, n_steps_checkpoint):
        """Run training."""

        average_loss = 0

        for i in xrange(1, n_steps + 1):
            batch_cues, batch_outcomes = get_events(cues, outcomes, probs, 1)

            loss_val, summaries = self.train_trial(batch_cues[0],
                                                   batch_outcomes[0])

            average_loss += loss_val

            if i % 500 == 0 and i:
                self.writer.add_summary(summaries, i)

            if i % 1000 == 0 and i:
                average_loss /= 1000
                print 'Average loss at step:', i, 'is', average_loss
                average_loss = 0

            if i % n_steps_checkpoint == 0 and i:
                print self.model_to_pd()
                self.save('checkpoints/%s_%i' % (self.name, i))

    def model_to_pd(self):
        """Transform embeddings to pandas data frame."""
        conn_matrix = self.session.run(self.embeddings)
        sorted_cues = [i[1] for i in sorted(self.cues_rev_dict.items())]
        sorted_outcomes = [i[1] for i in sorted(self.outcomes_rev_dict.items())]
        return pd.DataFrame(conn_matrix,
                            index=sorted_cues, columns=sorted_outcomes)

    def save(self, dirname, fname='model.ckpt'):
        """Save model."""
        self.saver = tf.train.Saver()

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self.saver.save(self.session, os.path.join(dirname, fname))

    def load(self, dirname, fname='model.ckpt'):
        """Load model."""
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, os.path.join(dirname, fname))
