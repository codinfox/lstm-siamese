from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import os

from lstm_siamese import dir_dictionary


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
def define_pretrain_input_single(input_directory, noise_std, epoch):
    # I will scan over every files in input_directory
    file_list = sorted(os.listdir(input_directory))
    file_list_mfcc = sorted([x for x in file_list if x.endswith('.tfrecords')])
    file_list_label = sorted([x for x in file_list if x.endswith('.tfrecords_label')])
    assert len(file_list_mfcc) == len(file_list_label)
    for f1, f2 in zip(file_list_mfcc, file_list_label):
        assert '/' not in f1 and os.path.splitext(f1)[0] == os.path.splitext(f2)[0]
    # build a filename queue

    rng_state = np.random.RandomState(seed=0)
    permutation = rng_state.permutation(len(file_list_mfcc))
    file_list_mfcc = np.array(file_list_mfcc)[permutation]
    file_list_label = np.array(file_list_label)[permutation]

    mfcc_file_name_producer = tf.train.string_input_producer([os.path.join(input_directory, x) for x in file_list_mfcc],
                                                             shuffle=False, num_epochs=epoch)
    label_file_name_producer = tf.train.string_input_producer([os.path.join(input_directory, x) for x in file_list_label],
                                                              shuffle=False, num_epochs=epoch)
    print('no shuffle')
    reader = tf.TFRecordReader()
    _, serialized_example_mfcc = reader.read(mfcc_file_name_producer)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example_mfcc,
                                                    context_features={
                                                        # 'label': tf.VarLenFeature(dtype=tf.int32),
                                                        'n_frame': tf.FixedLenFeature(dtype=tf.int64, shape=()),
                                                        'sentence_id': tf.FixedLenFeature(dtype=tf.int64, shape=()),
                                                    },
                                                    sequence_features={
                                                        'mfcc': tf.FixedLenSequenceFeature(shape=(26), dtype=tf.float32)
                                                    }
                                                    )

    reader2 = tf.TFRecordReader()
    _, serialized_example_label = reader2.read(label_file_name_producer)
    mfcc = sequence_parsed['mfcc']
    n_frame = tf.to_int32(context_parsed['n_frame'])
    # raw one for getting label in sparse form.
    print(mfcc, n_frame, serialized_example_label)
    return mfcc, n_frame, serialized_example_label


# def target_list_to_sparse_tensor(targetList):
#     '''make tensorflow SparseTensor from list of targets, with each element
#        in the list being a list or array with the values of the target sequence
#        (e.g., the integer values of a character map for an ASR target string)
#        See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
#        for example of SparseTensor format'''
#     indices = []
#     vals = []
#     for tI, target in enumerate(targetList):
#         for seqI, val in enumerate(target):
#             indices.append([tI, seqI])
#             vals.append(val)
#     shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
#     return (np.array(indices), np.array(vals), np.array(shape))


def define_pretrain_input_batch(input_directory, batch_size, noise_std=0.0, epoch=100):


    # load mean and std
    trainset_numpy = os.path.join(dir_dictionary['features'], 'TIMIT_train')
    mean_all = np.load(os.path.join(trainset_numpy, 'mean_legacy.npy'))
    std_all = np.load(os.path.join(trainset_numpy, 'std_legacy.npy'))



    mfcc, n_frame, serialized_example_label = define_pretrain_input_single(input_directory, noise_std=noise_std,
                                                                           epoch=epoch)
    # follow <https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html>
    mfcc_batch, n_frame_batch = tf.train.batch([mfcc, n_frame], batch_size=batch_size, shapes=[(None, 26),()],
                                               dynamic_pad=True)
    mfcc_batch = (mfcc_batch - mean_all)/std_all
    print('mean std done')

    raw_batch = tf.train.batch([serialized_example_label], batch_size=batch_size)
    label_sparse = tf.to_int32(tf.parse_example(raw_batch, {
        'label': tf.VarLenFeature(dtype=tf.int64),
    })['label'])
    # # then let's decode it.
    # mfcc_all = raw_decoded['mfcc']
    print('mfcc', mfcc_batch)
    print('frame', n_frame_batch)
    print('label', label_sparse)
    return mfcc_batch, n_frame_batch, label_sparse






# def define_input(nFeatures):
#     # batch size x max time x num features.
#     inputX = tf.placeholder(tf.float32, shape=(None, None, nFeatures))
#
#     targetIxs = tf.placeholder(tf.int64)
#     targetVals = tf.placeholder(tf.int32)
#     targetShape = tf.placeholder(tf.int64)
#     targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
#     seqLengths = tf.placeholder(tf.int32, shape=(None))
#
#     batch_size = tf.shape(inputX)[0]
#     max_timesteps = tf.shape(inputX)[1]
#
#     return (inputX, targetIxs, targetVals, targetShape), (targetY, seqLengths), (batch_size, max_timesteps)


def define_one_layer_BLSTM(inputX, seqLengths, nHidden):
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    print(inputX, seqLengths)
    (output_fw, output_bw), _ = bidirectional_dynamic_rnn(forwardH1, backwardH1, inputX, dtype=tf.float32,
                                                          sequence_length=seqLengths,
                                                          time_major=False)
    print(inputX)
    # both of shape (batch_size, max_time, hidden_size)
    output_combined = tf.concat(2, (output_fw, output_bw))
    # [num batch] x [num max time] x (hidden_sizex2)

    return output_combined, nHidden * 2  # hidden*2 is number of actual hidden states.

def define_one_layer_LSTM(inputX, seqLengths, nHidden):
    lstm_cell = rnn_cell.BasicLSTMCell(nHidden, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([lstm_cell]*2)

    initial = cell.zero_state(tf.shape(inputX)[0], tf.float32)


    outputs, _ = tf.nn.dynamic_rnn(cell, inputX, dtype=tf.float32,
                                    sequence_length=seqLengths, initial_state=initial,
                                    time_major=False)

    return outputs, nHidden  # hidden*2 is number of actual hidden states.


def define_logit_and_ctc(output_combined, targetY, seqLengths, nHiddenOutput, nClass):
    W = tf.Variable(tf.truncated_normal([nHiddenOutput, nClass],
                                        stddev=np.sqrt(2.0 / nHiddenOutput)))
    # Zero initialization
    # Tip: tf.zeros_initializer
    b = tf.Variable(tf.zeros([nClass]))
    batch_size = tf.shape(output_combined)[0]
    max_time = tf.shape(output_combined)[1]
    output_combined_reshape = tf.reshape(output_combined, [-1, nHiddenOutput])
    # Doing the affine projection
    logits = tf.matmul(output_combined_reshape, W) + b
    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_size, max_time, nClass])
    # Time major, this is convenient for edit distance.
    logits = tf.transpose(logits, (1, 0, 2))
    loss = ctc.ctc_loss(logits, targetY, seqLengths)
    cost = tf.reduce_mean(loss)

    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / tf.to_float(
        tf.size(targetY.values))

    return cost, errorRate, logits
