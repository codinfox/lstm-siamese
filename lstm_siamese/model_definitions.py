from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


# ####Learning Parameters
# learningRate = 0.001
# momentum = 0.9
# nEpochs = 120
# batchSize = 10
#
# ####Network Parameters
# nFeatures = 26  # 12 MFCC coefficients + energy, and derivatives
# nHidden = 128
# nClasses = 62  # 39 phonemes, plus the "blank" for CTC


def define_input(nFeatures):
    # batch size x max time x num features.
    inputX = tf.placeholder(tf.float32, shape=(None, None, nFeatures))

    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(None))

    batch_size = tf.shape(inputX)[0]
    max_timesteps = tf.shape(inputX)[1]

    return (inputX, targetIxs, targetVals, targetShape), (targetY, seqLengths), (batch_size, max_timesteps)


def define_one_layer_BLSTM(inputX, seqLengths, nHidden):
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    (output_fw, output_bw), _ = bidirectional_dynamic_rnn(forwardH1, backwardH1, inputX, dtype=tf.float32,
                                                          sequence_length=seqLengths,
                                                          time_major=False)
    # both of shape (batch_size, max_time, hidden_size)
    output_combined = tf.concat(2, (output_fw, output_bw))
    # [num batch] x [num max time] x (hidden_sizex2)

    return output_combined


def define_logit_and_ctc(output_combined, targetY, seqLengths, nHidden, nClass, batch_size):
    W = tf.Variable(tf.truncated_normal([nHidden, nClass],
                                        stddev=np.sqrt(2.0 / nHidden)))
    # Zero initialization
    # Tip: tf.zeros_initializer
    b = tf.Variable(tf.zeros([nClass]))
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

def define_siamese_loss(output_1, output_2, label, k=200, margin=5.0):
    o_1 = tf.slice(output_1, [0,0,0], [-1,k,-1])
    o_2 = tf.slice(output_2, [0,0,0], [-1,k,-1])
    o_1 = tf.reshape(o_1, [tf.shape(o_1)[0], -1])
    o_2 = tf.reshape(o_2, [tf.shape(o_2)[0], -1])
    labels_t = label
    labels_f = tf.sub(1.0, label, name="1-yi")          # labels_ = !labels;
    eucd2 = tf.pow(tf.sub(o_1, o_2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
    pos = tf.mul(labels_t, eucd2, name="yi_x_eucd2")
    # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
    # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
    neg = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss

def define_siamese_input(nFeatures):
    # batch size x max time x num features.
    inputX = tf.placeholder(tf.float32, shape=(None, None, nFeatures))

    seqLengths = tf.placeholder(tf.int32, shape=(None))
    return inputX, seqLengths
