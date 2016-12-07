from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import dir_dictionary
import os.path

from .model_definitions import define_one_layer_BLSTM, define_logit_and_ctc, \
    define_pretrain_input_batch, define_one_layer_LSTM


def run_one_model(params):
    assert set(params.keys()) == {
        'noise_std',
        'batch_size',
        'learning_rate',
        'n_hidden',
        'network_type'
    }
    # bidirection, one direction, double layer one direction
    assert params['network_type'] in {'BLSTM', 'LSTM', 'DLSTM'}
    # first, generate the training graph



def generate_training_graph():
    graph = tf.Graph()
    with graph.as_default():


def generate_test_graph():
    print('define test graph')
    graph = tf.Graph()
    with graph.as_default():



INPUT_PATH = os.path.join(dir_dictionary['features'], 'TIMIT_train_tf')
test_path = os.path.join(dir_dictionary['features'], 'TIMIT_test_tf')
####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 120
batchSize = 10

####Network Parameters
nFeatures = 26  # 12 MFCC coefficients + energy, and derivatives
nHidden = 128
nClasses = 62  # 39 phonemes, plus the "blank" for CTC

####Load data
# print('Loading data')
# batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)
# print('maxTimeSteps', maxTimeSteps, 'totalN', totalN)
# ####Define graph

batch_per_epoch = 3696 // batchSize
if batch_per_epoch * batchSize != 3696:
    batch_per_epoch += 1

print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    # (inputX, targetIxs, targetVals, targetShape), (targetY, seqLengths), (batch_size, max_timesteps) = define_input(
    #     nFeatures)
    mfcc_batch, n_frame_batch, label_sparse, name_batch = define_pretrain_input_batch(INPUT_PATH, batchSize, epoch=None)
    output_combined, nHidden_output = define_one_layer_BLSTM(mfcc_batch, n_frame_batch, nHidden)
    # output_combined, nHidden_output = define_one_layer_LSTM(mfcc_batch, n_frame_batch, nHidden)
    print(nHidden_output)
    loss, errorRate, logits = define_logit_and_ctc(output_combined, label_sparse, n_frame_batch, nHidden_output,
                                                   nClasses)
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    output_combined_shape = tf.shape(output_combined)
    logits_shape = tf.shape(logits)

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    mini_batch_current = 0
    try:
        while not coord.should_stop():
            # Run training steps or whatever
            _, l, er, output_shape, logits_shape_np, name_batch_np = session.run([optimizer, loss, errorRate,
                                                                                  output_combined_shape, logits_shape,
                                                                                  name_batch])
            print(output_shape, logits_shape_np)
            epoch_current = mini_batch_current / batch_per_epoch
            mini_batch_current += 1

            print('epoch {:.2f}/{}'.format(epoch_current, nEpochs))
            print('mini {}/{}'.format(mini_batch_current % batch_per_epoch, batch_per_epoch))
            print('loss:', l)
            print('error rate:', er)
            print(name_batch_np)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
