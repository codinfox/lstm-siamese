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
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
import numpy as np
from ctc_example.utils import load_batched_data
from lstm_siamese import dir_dictionary
import os.path

from lstm_siamese.model_definitions import define_input, define_one_layer_BLSTM, define_logit_and_ctc

trainset_numpy_feature = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'feature')
trainset_numpy_label = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'label')
INPUT_PATH = trainset_numpy_feature
TARGET_PATH = trainset_numpy_label

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
print('Loading data')
batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)
print('maxTimeSteps', maxTimeSteps, 'totalN', totalN)
####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    (inputX, targetIxs, targetVals, targetShape), (targetY, seqLengths), (batch_size, max_timesteps) = define_input(
        nFeatures)
    output_combined = define_one_layer_BLSTM(inputX, seqLengths, nHidden)
    loss, errorRate = define_logit_and_ctc(output_combined, targetY, seqLengths, nHidden, nClasses, batch_size)
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    output_combined_shape = tf.shape(output_combined)

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    for epoch in range(nEpochs):
        print('Epoch', epoch + 1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData))  # randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            batchInputs = batchInputs.transpose((1,0,2))
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, l, er, output_shape = session.run([optimizer, loss, errorRate, output_combined_shape], feed_dict=feedDict)
            print(output_shape)
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er * len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch + 1, 'error rate:', epochErrorRate)
