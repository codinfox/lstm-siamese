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
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from ctc_example.utils import load_batched_data
from lstm_siamese import dir_dictionary
import os.path

trainset_numpy_feature = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'feature')
trainset_numpy_label = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'label')
INPUT_PATH = trainset_numpy_feature
TARGET_PATH = trainset_numpy_label

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 120
batchSize = 64 

####Network Parameters
nFeatures = 26 #12 MFCC coefficients + energy, and derivatives
nHidden = 128
nClasses = 62 #39 phonemes, plus the "blank" for CTC
NLAYER = 2

####Load data
print('Loading data')
batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)
print('maxTimeSteps', maxTimeSteps, 'totalN', totalN)
####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow
        
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    ####Weights & biases

    W = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                    stddev=0.01))
    b = tf.Variable(tf.zeros([nClasses]))

    ####Network
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NLAYER)

    # Get lstm cell output
    outputs, _ = rnn.rnn(cell, inputList, dtype=tf.float32)
    
    outputs = tf.reshape(outputs, (-1, nHidden))
    logits = tf.matmul(outputs, W) + b
    
    logits = tf.reshape(logits, (batchSize, -1, nClasses))
    logits = tf.transpose(logits, [1,0,2])
    
    ####Optimizing
    loss = tf.reduce_mean(ctc.ctc_loss(logits, targetY, seqLengths))
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    ####Evaluating
    predictions = tf.to_int32(ctc.ctc_greedy_decoder(logits, seqLengths)[0][0])

    err = tf.edit_distance(predictions, targetY, normalize=True)
    err.set_shape([None])
    err = tf.reduce_mean(err, name='error')

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        if epoch % 10 == 0:
            print('Saving Graph')
            tf.train.Saver().save(session, "/home/zhihaol/807/model.ckpt")
            tf.train.write_graph(session.graph_def, "/home/zhihaol/807/", "model_graph.pbtxt", True)
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, l, er, lmt = session.run([optimizer, loss, err, logitsMaxTest], feed_dict=feedDict)
            print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er
        epochErrorRate = batchErrors.sum() / len(batchErrors)
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
