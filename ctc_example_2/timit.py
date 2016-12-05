#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: timit.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import os, sys
import argparse
from collections import Counter
import operator
import six
from six.moves import map, range

from tensorpack import *
from tensorpack.tfutils.gradproc import  *
from tensorpack.utils.lut import LookUpTable
from tensorpack.utils.globvars import globalns as param
import tensorpack.tfutils.symbolic_functions as symbf
from timitdata import TIMITData, TIMITBatch


BATCH = 64
NLAYER = 2
HIDDEN = 128
NR_CLASS =  26 + 1 + 1

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, None, 39], 'feat'),   # bxmaxseqx39
                InputVar(tf.int32, None, 'label', True),   #b x maxlen
                InputVar(tf.int32, [None], 'seqlen'),   # b
                ]

    def _build_graph(self, input_vars):
        feat, label, seqlen = input_vars

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * NLAYER)

        initial = cell.zero_state(tf.shape(feat)[0], tf.float32)

        outputs, last_state = tf.nn.dynamic_rnn(cell, feat,
                seqlen, initial,
                dtype=tf.float32, scope='rnn')

        # o: b x t x HIDDEN
        output = tf.reshape(outputs, [-1, HIDDEN])  # (Bxt) x rnnsize
        logits = FullyConnected('fc', output, NR_CLASS, nl=tf.identity,
                W_init=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.reshape(logits, (BATCH, -1, NR_CLASS))

        loss = tf.nn.ctc_loss(logits, label, seqlen, time_major=False)

        self.cost = tf.reduce_mean(loss, name='cost')
        #self.cost = symbf.print_stat(self.cost)

        logits = tf.transpose(logits, [1,0,2])
        predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(logits, seqlen)[0][0])
        err = tf.edit_distance(predictions, label, normalize=True)
        err.set_shape([None])
        err = tf.reduce_mean(err, name='error')
        #err = symbf.print_stat(err)
        summary.add_moving_summary(err)
        #print dec.name

    def get_gradient_processor(self):
        return [GlobalNormClip(5)]

def get_config():
    logger.auto_set_dir()

    ds = TIMITData()
    ds = TIMITBatch(ds, BATCH)
    ds = PrefetchDataZMQ(ds, 1)
    step_per_epoch = ds.size()

    lr = symbolic_functions.get_scalar_var('learning_rate', 1e-3, summary=True)

    return TrainConfig(
        dataset=ds,
        #optimizer=tf.train.AdamOptimizer(lr),
        optimizer=tf.train.MomentumOptimizer(lr, 0.9),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    SimpleTrainer(config).train()

