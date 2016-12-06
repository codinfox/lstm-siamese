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

from ctc_example.utils import load_batched_data_sentence
from lstm_siamese import dir_dictionary
import os.path

trainset_numpy_feature = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'feature')
trainset_numpy_label = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'label')
trainset_numpy_sentence = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'sentence_label')
INPUT_PATH = trainset_numpy_feature
TARGET_PATH = trainset_numpy_sentence

feature_list_left_all, feature_list_right_all, signal_list_all = load_batched_data_sentence(INPUT_PATH, TARGET_PATH)
print(feature_list_left_all[0].shape, feature_list_right_all[0].shape, signal_list_all[0])
print(signal_list_all.shape, signal_list_all.dtype)