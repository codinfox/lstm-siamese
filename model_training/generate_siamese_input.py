from __future__ import absolute_import, division, print_function

from lstm_siamese.timit_test_extract_feature import save_one_graph_features

# params_debug = {
#     'noise_std': 0.6,
#     'learning_rate': 0.001,
#     'n_hidden': 128,
#     'network_type': 'BLSTM1L',
#     'seed': 1,
#     'momentum': 0.9,
# }
#
# save_one_graph_features(params_debug, prefix='BLSTM1L_default_noise', start_epoch=42)

params_debug = {
    'noise_std': 0.0,
    'learning_rate': 0.001,
    'n_hidden': 128,
    'network_type': 'BLSTM1L',
    'seed': 1,
    'momentum': 0.9,
}

save_one_graph_features(params_debug, prefix='BLSTM1L_default_nonoise', start_epoch=42)

params_debug = {
    'noise_std': 0.6,
    'learning_rate': 0.001,
    'n_hidden': 128,
    'network_type': 'LSTM1L',
    'seed': 1,
    'momentum': 0.9,
}

save_one_graph_features(params_debug, prefix='LSTM1L_default_noise', start_epoch=49)

params_debug = {
    'noise_std': 0.6,
    'learning_rate': 0.001,
    'n_hidden': 128,
    'network_type': 'LSTM2L',
    'seed': 1,
    'momentum': 0.9,
}

# run_one_model(params_debug, prefix='aaa', start_epoch=2)
save_one_graph_features(params_debug, prefix='LSTM2L_default_noise', start_epoch=27)
