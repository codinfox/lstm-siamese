from __future__ import absolute_import, division, print_function

from lstm_siamese.timit_training_general import run_one_model

params_debug = {
    'noise_std': 0.6,
    'batch_size_train': 33,
    'batch_size_test_list': [66*2, 32*6],
    'learning_rate': 0.001,
    'n_hidden': 128,
    'network_type': 'BLSTM1L',
    'train_set': 'timit_train',
    'test_set_list': ['timit_train', 'timit_test'],
    'seed': 1,
    'momentum': 0.9,
}

#run_one_model(params_debug, prefix='aaa', start_epoch=2)
run_one_model(params_debug, prefix='BLSTM1L_default_noise', start_epoch=None)
