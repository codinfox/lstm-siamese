from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import dir_dictionary
import os.path
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import h5py

from .model_definitions import define_logit_and_ctc, \
    define_pretrain_input_batch, layer_func_dict

train_path = os.path.join(dir_dictionary['features'], 'TIMIT_train_tf')
test_path = os.path.join(dir_dictionary['features'], 'TIMIT_test_tf')
save_path = os.path.join(dir_dictionary['models'])

# http://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# batch_size_dict = {
#     train_path: 33,  # 3696 ones 3696/33 = 112 mini batch
#     test_path: 32, # 1344 ones 1344/32 = 42 mini batch
# }
dataset_mapping = {
    'timit_train': train_path,
    'timit_test': test_path,
    # 'test_core': None
}

set_size_dict = {
    'timit_train': 3696,  # 3696 ones 3696/33 = 112 mini batch
    'timit_test': 1344,  # 1344 ones 1344/32 = 42 mini batch
    # 'test_core':
}


def _create_model_dir(save_path, prefix):
    model_dir = os.path.join(save_path, 'TIMIT_train_{}'.format(prefix))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir


def get_previous_state_file(prefix, epoch):
    model_dir = _create_model_dir(save_path, prefix)
    return os.path.join(model_dir, '{}.checkpoint'.format(epoch))


def get_stats_file(prefix):
    model_dir = _create_model_dir(save_path, prefix)
    return os.path.join(model_dir, 'stats.hdf5')


#
# 'input_mfcc': mfcc_batch,
#         'input_nframe': n_frame_batch,
#         'input_label': label_sparse,
#         'input_name': name_batch,
#         'rnn_output': output_combined,
#         'loss_individual': loss_individual,
#         'error_rate_this_batch': errorRate_this_batch,
#         'error_rate_raw': errorRate_raw,
#         'z_count_this_batch': z_count_this,
#         'loss_overall': loss_overall,
#         'logits': logits,
#         'predictions': predictions,

def test_one_graph(graph_def, saver_file):
    (train_graph, train_graph_variable), batch_per_epoch = graph_def

    with tf.Session(graph=train_graph, config=config) as session:
        # consider whether starting from previous saved states or not.

        loss_individual, error_rate_raw, z_count_this_batch, name_batch = (train_graph_variable['loss_individual'],
                                                                           train_graph_variable['error_rate_raw'],
                                                                           train_graph_variable['z_count_this_batch'],
                                                                           train_graph_variable['input_name'])
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        saver.restore(session, saver_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        loss_individual_all = []
        error_rate_raw_all = []
        z_all = []

        for mini_batch in range(batch_per_epoch):
            assert not coord.should_stop()  # I just ignore this
            # Run training steps or whatever
            (loss_individual_np,
             error_rate_raw_np, z_np, names_np) = session.run(
                [loss_individual, error_rate_raw, z_count_this_batch, name_batch])
            assert np.isscalar(error_rate_raw_np)
            assert np.isscalar(z_np)
            assert loss_individual_np.ndim == 1
            loss_individual_all.append(loss_individual_np)
            error_rate_raw_all.append(error_rate_raw_np)
            z_all.append(z_np)
            print(mini_batch + 1, batch_per_epoch)
            print(names_np[:5])
            print(names_np.shape)
        z_all = np.array(z_all).sum()
        error_rate_all = np.array(error_rate_raw_all).sum()
        loss_individual_all = np.concatenate(loss_individual_all)
        print(loss_individual_all.shape)
        mean_loss = loss_individual_all.mean()
        mean_error_rate = error_rate_all / z_all
        print('mean loss', mean_loss, 'mean label error rate', error_rate_all / z_all)

        # seems this is clean
        coord.request_stop()

    return mean_loss, mean_error_rate


def run_one_model(params, start_epoch=None, prefix=None, epoch=200):
    assert prefix is not None
    assert set(params.keys()) == {
        'noise_std',
        'batch_size_train',
        'batch_size_test_list',
        'learning_rate',
        'n_hidden',
        'network_type',
        'train_set',
        'test_set_list',
        'seed',
        'momentum',
    }
    # bidirection, one direction, double layer one direction
    assert params['network_type'] in {'BLSTM1L', 'LSTM1L', 'LSTM2L'}
    # first, generate the training graph
    (train_graph, train_graph_variable), batch_per_epoch = define_one_graph_wrapper(params['train_set'],
                                                                                    params['batch_size_train'],
                                                                                    deepcopy(params),
                                                                                    define_one_train_graph)
    batch_size_test_list = params['batch_size_test_list']
    test_set_list = params['test_set_list']
    assert len(batch_size_test_list) == len(test_set_list)
    assert len(test_set_list) >= 0
    test_graph_dict = OrderedDict()
    for test_set_name, test_set_batch_size in zip(test_set_list, batch_size_test_list):
        test_graph_dict[test_set_name] = define_one_graph_wrapper(test_set_name,
                                                                  test_set_batch_size,
                                                                  deepcopy(params), define_one_test_graph)

    with tf.Session(graph=train_graph, config=config) as session:
        # consider whether starting from previous saved states or not.

        optimizer, loss, error_rate, name_batch = (train_graph_variable['optimizer'],
                                                   train_graph_variable['loss_overall'],
                                                   train_graph_variable['error_rate_this_batch'],
                                                   train_graph_variable['input_name'])
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)

        if start_epoch is None:
            print('init')
            tf.initialize_all_variables().run()
            current_epoch = 0  # current means how many I have finished.
        else:
            # start_from should be 1, 2, 3, ....
            assert start_epoch > 0
            current_epoch = start_epoch
            # then load previous file
            saver.restore(session, get_previous_state_file(prefix, current_epoch))

        # check the HDF5.
        # load the file and then restore.
        # I will check that all the states were there. so anything smaller than start_epoch should be there.
        stats_file = get_stats_file(prefix)
        correct_group_list = [str(x) for x in range(current_epoch)]
        with h5py.File(stats_file) as f_stat:
            assert set(f_stat.keys()) == set(correct_group_list), 'wrong start_epoch'

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        while current_epoch < epoch:
            loss_per_batch = []
            error_per_batch = []
            for mini_batch in range(batch_per_epoch):
                assert not coord.should_stop()  # I just ignore this
                # Run training steps or whatever
                _, l, er, name_batch_np = session.run([optimizer, loss, error_rate, name_batch])

                print('epoch {}, batch {}/{}'.format(current_epoch + 1, mini_batch + 1, batch_per_epoch))
                print('loss:', l)
                print('error rate:', er)
                assert np.isscalar(l) and np.isscalar(er)
                loss_per_batch.append(l)
                error_per_batch.append(er)
                # print(name_batch_np)
            loss_per_batch = np.array(loss_per_batch)
            error_per_batch = np.array(error_per_batch)
            this_group = str(current_epoch)

            current_epoch += 1
            # save current state.
            print('saving epoch {}'.format(current_epoch))
            # here 1-index is used.
            current_saver_file = get_previous_state_file(prefix, current_epoch)
            saver.save(session, current_saver_file)

            with h5py.File(stats_file) as f_stat:
                grp_this = f_stat.create_group(this_group)
                grp_this.create_dataset('loss_per_batch', data=loss_per_batch)
                grp_this.create_dataset('error_per_batch', data=error_per_batch)
                grp_this.create_group('test')
                grp_this.file.flush()

            # then perform test.
            for test_set in test_graph_dict:
                print('test', test_set)
                (mean_loss, mean_error_rate) = test_one_graph(test_graph_dict[test_set], current_saver_file)
                with h5py.File(stats_file) as f_stat:
                    grp_this_test = f_stat[this_group + '/test'].create_group(test_set)
                    grp_this_test.attrs['mean_loss'] = mean_loss
                    grp_this_test.attrs['mean_error_rate'] = mean_error_rate
                    grp_this_test.file.flush()

        # seems this is clean
        coord.request_stop()


def define_one_graph_wrapper(train_set, batch_size_train,
                             params_train, graph_define_func):
    train_set_size = set_size_dict[train_set]
    batch_per_epoch = (train_set_size // batch_size_train)
    assert batch_per_epoch * batch_size_train == train_set_size, 'epoch size not good!'
    assert 'batch_size' not in params_train
    params_train['batch_size'] = batch_size_train
    train_graph_variable = graph_define_func(params_train, dataset_mapping[train_set])
    return train_graph_variable, batch_per_epoch


def define_one_train_graph(params, train_dir):
    print(params.keys())
    assert set(params.keys()) >= {'noise_std',
                                  'batch_size',
                                  'learning_rate',
                                  'momentum',
                                  'n_hidden',
                                  'network_type',
                                  'seed'
                                  }

    network_type = params['network_type']
    # first define input
    batchSize = params['batch_size']
    nHidden = params['n_hidden']
    noise_std = params['noise_std']
    momentum = params['momentum']
    learningRate = params['learning_rate']
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(params['seed'])
        var_dict = define_input_and_loss_graph(network_type, train_dir, nHidden, batchSize, noise_std=noise_std)
        # then add optimizer
        optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(var_dict['loss_overall'])
        assert 'optimizer' not in var_dict
        var_dict['optimizer'] = optimizer

    return graph, var_dict


def define_one_test_graph(params, test_dir):
    assert set(params.keys()) >= {
        'n_hidden',
        'network_type',
        'batch_size'
    }
    network_type = params['network_type']
    batchSize = params['batch_size']
    nHidden = params['n_hidden']

    graph = tf.Graph()
    with graph.as_default():
        # no seed needed.
        var_dict = define_input_and_loss_graph(network_type, test_dir, nHidden, batchSize, noise_std=0.0)
        # then add optimizer

    return graph, var_dict


def define_input_and_loss_graph(network_type, input_dir, nHidden, batchSize, noise_std=0.0):
    nClasses = 62  # 61 TIMIT + 1 blank
    print('noise', noise_std)
    func_for_rnn_layer = layer_func_dict[network_type]
    mfcc_batch, n_frame_batch, label_sparse, name_batch = define_pretrain_input_batch(input_dir, batchSize,
                                                                                      epoch=None, noise_std=noise_std)
    output_combined, nHidden_output = func_for_rnn_layer(mfcc_batch, n_frame_batch, nHidden)
    ((loss_overall, loss_individual),
     (errorRate_raw, z_count_this, errorRate_this_batch),
     (logits, predictions)) = define_logit_and_ctc(output_combined, label_sparse,
                                                   n_frame_batch, nHidden_output,
                                                   nClasses)
    return {
        'input_mfcc': mfcc_batch,
        'input_nframe': n_frame_batch,
        'input_label': label_sparse,
        'input_name': name_batch,
        'rnn_output': output_combined,
        'loss_individual': loss_individual,
        'error_rate_this_batch': errorRate_this_batch,
        'error_rate_raw': errorRate_raw,
        'z_count_this_batch': z_count_this,
        'loss_overall': loss_overall,
        'logits': logits,
        'predictions': predictions,
    }
