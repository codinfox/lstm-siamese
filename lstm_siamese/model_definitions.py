from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.python.ops import functional_ops
import os

from lstm_siamese import dir_dictionary


# essentially from https://github.com/tensorflow/tensorflow/issues/1742
def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = tf.shape(labels)
    num_batches_tns = tf.pack([label_shape[0]])
    max_num_labels_tns = tf.pack([label_shape[1]])

    def range_less_than(_, current_input):
        return tf.range(label_shape[1]) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    dense_mask = functional_ops.scan(range_less_than, label_lengths, initializer=init,
                                     parallel_iterations=1)
    # dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns),
                             label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
                                          tf.reverse(label_shape, [True])))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)

    indices = tf.transpose(tf.reshape(tf.concat(0, [batch_ind, label_ind]), [2, -1]))
    vals_sparse = tf.gather_nd(labels, indices)
    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))


def define_pretrain_input_single(input_directory, epoch):
    # I will scan over every files in input_directory
    file_list = sorted(os.listdir(input_directory))
    for f in file_list:
        assert f.endswith('.tfrecords')
    # file_list_mfcc = sorted([x for x in file_list if x.endswith('.tfrecords')])
    # file_list_label = sorted([x for x in file_list if x.endswith('.tfrecords_label')])
    # assert len(file_list_mfcc) == len(file_list_label)
    # for f1, f2 in zip(file_list_mfcc, file_list_label):
    #     assert '/' not in f1 and os.path.splitext(f1)[0] == os.path.splitext(f2)[0]
    # # build a filename queue
    # rng_state = np.random.RandomState(seed=0)
    # permutation = rng_state.permutation(len(file_list))
    # file_list = np.array(file_list)[permutation]
    # rng_state = np.random.RandomState(seed=0)
    # permutation = rng_state.permutation(len(file_list_mfcc))
    # file_list_mfcc = np.array(file_list_mfcc)[permutation]
    # file_list_label = np.array(file_list_label)[permutation]

    file_name_producer = tf.train.string_input_producer([os.path.join(input_directory, x) for x in file_list],
                                                        shuffle=True, num_epochs=epoch)
    # label_file_name_producer = tf.train.string_input_producer(
    #     [os.path.join(input_directory, x) for x in file_list_label],
    #     shuffle=True, num_epochs=epoch, seed=0)
    reader = tf.TFRecordReader()
    _, serialized_example_mfcc = reader.read(file_name_producer)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example_mfcc,
                                                                       context_features={
                                                                           # 'label': tf.VarLenFeature(dtype=tf.int32),
                                                                           'n_frame': tf.FixedLenFeature(dtype=tf.int64,
                                                                                                         shape=()),
                                                                           'sentence_id': tf.FixedLenFeature(
                                                                               dtype=tf.int64, shape=()),
                                                                           'name': tf.FixedLenFeature(
                                                                               dtype=tf.string, shape=()),
                                                                       },
                                                                       sequence_features={
                                                                           'mfcc': tf.FixedLenSequenceFeature(
                                                                               shape=(26), dtype=tf.float32),
                                                                           'label': tf.FixedLenSequenceFeature(
                                                                               shape=(), dtype=tf.int64),
                                                                       }
                                                                       )

    # reader2 = tf.TFRecordReader()
    # _, serialized_example_label = reader2.read(label_file_name_producer)
    mfcc = sequence_parsed['mfcc']
    n_frame = tf.to_int32(context_parsed['n_frame'])
    label = sequence_parsed['label']
    label_len = tf.shape(label)[0]
    sentence_id = context_parsed['sentence_id']
    name = context_parsed['name']
    # raw one for getting label in sparse form.
    # print(mfcc, n_frame, label, sentence_id, name)
    return mfcc, n_frame, label, label_len, sentence_id, name


def define_pretrain_input_batch(input_directory, batch_size, noise_std=0.0, epoch=100):
    # load mean and std
    trainset_numpy = os.path.join(dir_dictionary['features'], 'TIMIT_train')
    mean_all = np.load(os.path.join(trainset_numpy, 'mean.npy'))
    std_all = np.load(os.path.join(trainset_numpy, 'std.npy'))

    mfcc, n_frame, label, label_len, _, name = define_pretrain_input_single(input_directory,
                                                                            epoch=epoch)
    # follow <https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html>
    mfcc_batch, n_frame_batch, label_batch, label_len_batch, name_batch = tf.train.batch(
        [mfcc, n_frame, label, label_len, name],
        batch_size=batch_size,
        shapes=[(None, 26), (),
                (None,), (), ()],
        dynamic_pad=True,
        allow_smaller_final_batch=False)
    mfcc_batch = (mfcc_batch - mean_all) / std_all
    if noise_std != 0:
        mfcc_batch = mfcc_batch + tf.random_normal(tf.shape(mfcc_batch), stddev=noise_std)
    print('mean std done')
    # convert to sparse
    label_sparse = tf.to_int32(ctc_label_dense_to_sparse(label_batch, label_len_batch))
    # print('mfcc', mfcc_batch)
    # print('frame', n_frame_batch)
    # print('label', label_sparse)
    # print('name', name_batch)
    # print('label_dense', label_batch)
    return mfcc_batch, n_frame_batch, label_sparse, name_batch


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
    # print(inputX, seqLengths)
    (output_fw, output_bw), _ = bidirectional_dynamic_rnn(forwardH1, backwardH1, inputX, dtype=tf.float32,
                                                          sequence_length=seqLengths,
                                                          time_major=False)
    # print(inputX)
    # both of shape (batch_size, max_time, hidden_size)
    output_combined = tf.concat(2, (output_fw, output_bw))
    # [num batch] x [num max time] x (hidden_sizex2)

    return output_combined, nHidden * 2  # hidden*2 is number of actual hidden states.


def define_two_layer_LSTM(inputX, seqLengths, nHidden):
    lstm_cell = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * 2)

    initial = cell.zero_state(tf.shape(inputX)[0], tf.float32)

    outputs, _ = dynamic_rnn(cell, inputX, dtype=tf.float32,
                             sequence_length=seqLengths, initial_state=initial,
                             time_major=False)

    return outputs, nHidden  # hidden*2 is number of actual hidden states.


def define_one_layer_LSTM(inputX, seqLengths, nHidden):
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    print(inputX, seqLengths)
    initial = forwardH1.zero_state(tf.shape(inputX)[0], tf.float32)
    outputs, _ = dynamic_rnn(forwardH1, inputX, dtype=tf.float32,
                             sequence_length=seqLengths,
                             time_major=False)

    return outputs, nHidden


layer_func_dict = {
    'BLSTM1L': define_one_layer_BLSTM,
    'LSTM2L': define_two_layer_LSTM,
    'LSTM1L': define_one_layer_LSTM,
}


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
    loss_individual = ctc.ctc_loss(logits, targetY, seqLengths)
    loss_overall = tf.reduce_mean(loss_individual)

    # just use this beam search.
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits, seqLengths)[0][0])
    errorRate_raw = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False))
    z_count_this = tf.size(targetY.values)
    errorRate_this_batch = errorRate_raw / tf.to_float(z_count_this)

    return (loss_overall, loss_individual), (errorRate_raw, z_count_this, errorRate_this_batch), (logits, predictions)
