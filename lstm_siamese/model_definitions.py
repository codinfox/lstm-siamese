from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 120
batchSize = 10

####Network Parameters
nFeatures = 26  # 12 MFCC coefficients + energy, and derivatives
nHidden = 128
nClasses = 62  # 39 phonemes, plus the "blank" for CTC


def define_input(nFeatures):
    # batch size x max time x num features.
    inputX = tf.placeholder(tf.float32, shape=(None, None, nFeatures))

    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    batch_size = tf.shape(inputX)[0]
    max_timesteps = tf.shape(inputX)[1]

    return (inputX, targetIxs, targetVals, targetShape), (targetY, seqLengths), (batch_size, max_timesteps)


def define_one_layer_BLSTM(inputX, seqLengths, nHidden):
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    (output_fw, output_bw), _ = bidirectional_dynamic_rnn(forwardH1, backwardH1, inputX, dtype=tf.float32,
                                                          scope='BDLSTM_H1', sequence_length=seqLengths,
                                                          time_major=False)
    # both of shape (batch_size, max_time, hidden_size)
    output_combined = tf.concat(2, (output_fw, output_bw))
    # [num batch] x [num max time] x (hidden_sizex2)

    return output_combined


def define_logit_and_ctc(output_combined, targetY, seqLengths, nHidden, nClass, batch_size):
    W = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    # Zero initialization
    # Tip: tf.zeros_initializer
    b = tf.Variable(tf.zeros([nClasses]))
    output_combined_reshape = tf.reshape(output_combined, [-1, nHidden])
    # Doing the affine projection
    logits = tf.matmul(output_combined_reshape, W) + b
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, -1, nClass])
    # Time major, this is convenient for edit distance.
    logits = tf.transpose(logits, (1, 0, 2))
    loss = ctc.ctc_loss(logits, targetY, seqLengths)
    cost = tf.reduce_mean(loss)

    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / tf.to_float(
        tf.size(targetY.values))

    return cost, errorRate
