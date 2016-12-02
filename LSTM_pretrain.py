
# coding: utf-8

# In[1]:

'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict character sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is test code to run on the
8-item data set in the "sample_data" directory, for those without access to TIMIT.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from train_utils import load_batched_data

INPUT_PATH = '/home/zhihaol/807/TIMIT/train/mfcc/' #directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = '/home/zhihaol/807/TIMIT/train/txt/' #directory of nCharacters 1-D array .npy files

####Learning Parameters
learningRate = 0.01
momentum = 0.9
nEpochs = 150
batchSize = 64

####Network Parameters
nFeatures = 39 #12 MFCC coefficients + energy, and derivatives
nHidden = 64
# nClasses = 62#27 characters, plus the "blank" for CTC
nClasses = ord('z') - ord('a') + 1 + 1 + 1
####Load data
print('Loading data')
batchedData, maxTimeSteps, totalN = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)


# In[ ]:

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():
        
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=[None, None, nFeatures])
    
    inputXrs = tf.transpose(inputX, [1, 0, 2])

    targetY = tf.sparse_placeholder(tf.int32)
    
    seqLengths = tf.placeholder(tf.int32, shape=[None])

    ####Weights & biases
    
    W = tf.Variable(tf.truncated_normal([nHidden,
                                         nClasses],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[nClasses]))

    ####Network
    
    cell = tf.nn.rnn_cell.LSTMCell(nHidden, state_is_tuple=True)

    # Stacking rnn cells
    stack = tf.nn.rnn_cell.MultiRNNCell([cell] * 1,
                                        state_is_tuple=True)
    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputXrs, seqLengths, dtype=tf.float32)
    

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, nHidden])
    
    logits = tf.matmul(outputs, W) + b
    
#     Reshaping back to the original shape
    shape = tf.shape(inputXrs)
    batch_s, max_timesteps = shape[0], shape[1]
    logits = tf.reshape(logits, [batch_s, -1, nClasses])
    # Time major
    logits = tf.transpose(logits, (1, 0, 2))


    loss = ctc.ctc_loss(logits, targetY, seqLengths)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learningRate,
                                           momentum).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = ctc.ctc_greedy_decoder(logits, seqLengths)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targetY))


# In[ ]:

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        
        for batch, batchOrigI in enumerate(batchRandIxs):
            batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
            
#             print(batchInputs)
            
            feedDict = {inputX: batchInputs,
                        targetY: batchTargetSparse,
                        seqLengths: batchSeqLengths}
            
            _, l, er = session.run([optimizer, cost, ler], feed_dict=feedDict)
#             this is how we get ctc decoded label
#             d = session.run(decoded, feed_dict=feedDict)
#             print(d)
            
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er*len(batchSeqLengths)
        print(batchErrors)
        epochErrorRate = batchErrors.sum() / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)

