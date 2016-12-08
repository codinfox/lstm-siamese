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

from .timit_training_general import (config,
                                     define_one_test_graph,
                                     define_one_graph_wrapper, get_previous_state_file,
                                     _create_model_dir, save_path)


# 'rnn_output': output_combined,
# 'loss_individual': loss_individual,
# 'error_rate_this_batch': errorRate_this_batch,
# 'error_rate_raw': errorRate_raw,
# 'z_count_this_batch': z_count_this,
# 'loss_overall': loss_overall,
# 'logits': logits,
# 'predictions': predictions,

def get_hdf5_file(prefix):
    model_dir = _create_model_dir(save_path, prefix)
    return os.path.join(model_dir, 'predictions.hdf5')

def save_to_hdf5(prefix, all_data):
    name_all, prediction_all, output_all, n_frame_all = all_data
    assert len(name_all) == len(prediction_all)
    file_name = get_hdf5_file(prefix)
    with h5py.File(file_name) as f_out:
        for names, predictions in zip(name_all, prediction_all):
            shape = predictions.shape
            assert type(predictions.indices) == type(predictions.values) == np.ndarray
            assert shape[0] == len(names)
            for i, name_this in enumerate(names):
                mask_this = (predictions.indices[:,0] == i)
                column_this = predictions.indices[:,1][mask_this]
                assert np.array_equal(np.sort(column_this), column_this)
                values_this = predictions.values[mask_this].ravel()
                print(values_this)
                group_to_save = '/'.join(name_this.split('_'))
                if name_this not in f_out:
                    print('saving {}'.format(group_to_save))
                    f_out.create_dataset(group_to_save, data=values_this)





def save_one_graph_features(params, start_epoch=None, prefix=None):
    assert prefix is not None
    assert start_epoch is not None
    assert set(params.keys()) == {
        'noise_std',
        'learning_rate',
        'n_hidden',
        'network_type',
        'seed',
        'momentum',
    }
    # bidirection, one direction, double layer one direction
    assert params['network_type'] in {'BLSTM1L', 'LSTM1L', 'LSTM2L'}
    # first, generate the training graph
    # trainset_graph_all = define_one_graph_wrapper('timit_train',
    #                                               66 * 2,
    #                                               deepcopy(params),
    #                                               define_one_test_graph)
    testset_graph_all = define_one_graph_wrapper('timit_test',
                                                 32 * 6,
                                                 deepcopy(params),
                                                 define_one_test_graph)
    current_saver_file = get_previous_state_file(prefix, start_epoch)
    # name_all, prediction_all, output_all, n_frame_all = test_one_graph_extended(trainset_graph_all, current_saver_file)
    all_data = test_one_graph_extended(testset_graph_all, current_saver_file)
    save_to_hdf5(prefix, all_data)


def test_one_graph_extended(graph_def, saver_file):
    (train_graph, train_graph_variable), batch_per_epoch = graph_def

    with tf.Session(graph=train_graph, config=config) as session:
        # consider whether starting from previous saved states or not.

        (name_batch, prediction_batch,
         output_batch, n_frame_batch) = (train_graph_variable['input_name'], train_graph_variable['predictions'],
                                         train_graph_variable['rnn_output'], train_graph_variable['input_nframe'])
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        saver.restore(session, saver_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        name_all = []
        prediction_all = []
        output_all = []
        n_frame_all = []

        for mini_batch in range(batch_per_epoch):
            assert not coord.should_stop()  # I just ignore this
            # Run training steps or whatever
            (names_np, prediction_tfvalue,
             output_np, n_frame_np) = session.run(
                [name_batch, prediction_batch,
                 output_batch, n_frame_batch])
            name_all.append(names_np)
            prediction_all.append(prediction_tfvalue)
            output_all.append(output_np)
            n_frame_all.append(n_frame_np)
            print(mini_batch + 1, batch_per_epoch)
            print(output_np.shape, n_frame_np.shape)
            print(prediction_tfvalue.shape)
            print(names_np[:5])
        name_all_all = np.concatenate(name_all)
        assert len(np.unique(name_all_all)) == len(name_all_all)
        print(name_all_all.shape)

        # then I should save everything.


        # seems this is clean
        coord.request_stop()

        return name_all, prediction_all, output_all, n_frame_all
