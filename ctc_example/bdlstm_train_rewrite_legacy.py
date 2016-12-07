'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict phoneme sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is basically a recreation of an experiment
on the TIMIT data set from chapter 7 of Alex Graves's book (Graves, Alex. Supervised Sequence 
Labelling with Recurrent Neural Networks, volume 385 of Studies in Computational Intelligence.
Springer, 2012.), minus the early stopping.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lstm_siamese import dir_dictionary
import os.path

from lstm_siamese.model_definitions_legacy import define_one_layer_BLSTM, define_logit_and_ctc, \
    define_pretrain_input_batch, define_one_layer_LSTM

# trainset_numpy_feature = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'feature')
# trainset_numpy_label = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'label')
# INPUT_PATH = trainset_numpy_feature
# TARGET_PATH = trainset_numpy_label
INPUT_PATH = os.path.join(dir_dictionary['features'], 'TIMIT_train_tf_legacy')
####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 120
batchSize = 2
batchSize = 33

####Network Parameters
nFeatures = 26  # 12 MFCC coefficients + energy, and derivatives
nHidden = 128
nClasses = 62  # 39 phonemes, plus the "blank" for CTC

####Load data
# print('Loading data')
# batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)
# print('maxTimeSteps', maxTimeSteps, 'totalN', totalN)
# ####Define graph

batch_per_epoch = 3696//batchSize

print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(1)
    # (inputX, targetIxs, targetVals, targetShape), (targetY, seqLengths), (batch_size, max_timesteps) = define_input(
    #     nFeatures)
    mfcc_batch, n_frame_batch, label_sparse = define_pretrain_input_batch(INPUT_PATH, batchSize, epoch=None)
    output_combined, nHidden_output = define_one_layer_BLSTM(mfcc_batch, n_frame_batch, nHidden)
    #output_combined, nHidden_output = define_one_layer_LSTM(mfcc_batch, n_frame_batch, nHidden)
    print(nHidden_output)
    loss, errorRate, logits = define_logit_and_ctc(output_combined, label_sparse, n_frame_batch, nHidden_output,
                                                   nClasses)
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    output_combined_shape = tf.shape(output_combined)
    logits_shape = tf.shape(logits)

    label_sparse_dense = tf.sparse_tensor_to_dense(label_sparse)


    print(mfcc_batch, n_frame_batch, label_sparse)
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
            _, l, er, output_shape, logits_shape_np, mfcc_batch_np,\
            label_sparse_dense_np, n_frame_batch_np, label_sparse_np = session.run([optimizer, loss, errorRate,
                                                                   output_combined_shape, logits_shape, mfcc_batch,
                                                                   label_sparse_dense, n_frame_batch, label_sparse])
            # l, er, output_shape, logits_shape_np, mfcc_batch_np = session.run([loss, errorRate,
            #                                                                       output_combined_shape, logits_shape,
            #                                                                       mfcc_batch])
            print(output_shape, logits_shape_np)
            epoch_current = mini_batch_current / batch_per_epoch
            mini_batch_current += 1

            print('epoch {:.2f}/{}'.format(epoch_current, nEpochs))
            print('mini {}/{}'.format(mini_batch_current % batch_per_epoch, batch_per_epoch))
            print('loss:', l)
            print('error rate:', er)
            # print('mean {} std {}'.format(mfcc_batch_np.mean(),mfcc_batch_np.std()))
            # print('label mean {} std {}'.format(label_sparse_dense_np.mean(), label_sparse_dense_np.std()))
            # print(label_sparse_np)
            # print(n_frame_batch_np)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
