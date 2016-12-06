"""this file generates MFCC features for TIMIT set
it will keep convert all sentences into raw MFCC_0_D features,
and will map 61 phonemes to 0-60.

Notice that in training, SA sentences should be ignored, which is what I found by googling.
"""

# no unicode for best compatibility
from __future__ import division, print_function, absolute_import
import os.path
import h5py
import numpy as np
from lstm_siamese import dir_dictionary, htkmfc
import subprocess
from tempfile import NamedTemporaryFile
import tensorflow as tf

mfcc_config_file = os.path.join(dir_dictionary['root'], 'feature_generation', 'mfcc_config')
hcopy_bin = os.path.join(dir_dictionary['root'], '3rdparty', 'htk', 'bin.cpu', 'HCopy')

# in total, 61
_phoneme_sequence = [
    'iy', 'ix', 'aa', 'ch', 'zh',
    'eh', 'el', 'ah', 'ow', 'ao',
    'ih', 'tcl', 'en', 'ey', 'aw',
    'h#', 'ay', 'ax', 'er', 'pau',
    'eng', 'gcl', 'ng', 'nx', 'r',
    'pcl', 't', 'bcl', 'dcl', 'th',
    'dh', 'kcl', 'v', 'hv', 'oy',
    'hh', 'jh', 'dx', 'ax-h', 'em',
    'd', 'axr', 'b', 'ux', 'g',
    'f', 'uw', 'm', 'l', 'n',
    'q', 'p', 's', 'sh', 'uh',
    'w', 'epi', 'y', 'ae', 'z',
    'k'
]
assert len(_phoneme_sequence) == 61
_phoneme_mapping = dict(zip(_phoneme_sequence, range(61)))
print(_phoneme_mapping)


def convert_one_feature(infile, outfile):
    assert os.path.isabs(infile)
    # call hcopy to convert it
    subprocess.check_output([hcopy_bin, '-C', mfcc_config_file, infile, outfile])
    # then use htkmfc to read it
    read_obj = htkmfc.HTKFeat_read()
    read_obj.open(outfile)
    feature = read_obj.getall()
    read_obj.fh.close()
    assert feature.dtype == np.float32 and feature.ndim == 2 and feature.shape[1] == 26
    print(feature.shape)
    return feature


def get_phoneme_label(infile):
    # given a PHN file, get the phoneme sequence in uint8 integers.
    # read the file, get the third column
    temp_phones = np.loadtxt(infile, dtype={'names': ('start', 'end', 'phone'),
                                            'formats': (np.int32, np.int32, 'S4')},
                             comments='')['phone']
    temp_phones_index = np.asarray([_phoneme_mapping[x] for x in temp_phones])
    assert np.all(temp_phones_index < 61) and np.all(temp_phones_index >= 0)
    temp_phones_index_converted = temp_phones_index.astype(np.uint8)
    assert np.array_equal(temp_phones_index, temp_phones_index_converted)
    assert temp_phones_index_converted.ndim == 1

    return temp_phones_index_converted


all_phoneme_set = {'set': set()}


def main_loop(start_dir, grp):
    # get a unique file name
    tmp_mfc_file_ptr = NamedTemporaryFile(delete=False)
    tmp_mfc_file = tmp_mfc_file_ptr.name
    tmp_mfc_file_ptr.close()
    for subdir, dirs, files in os.walk(start_dir):
        for f in files:
            if f.endswith(".PHN"):
                grp_to_save = subdir.split('/')[-3:]
                utterance_id = os.path.splitext(f)[0]
                assert utterance_id[:2] in {'SA', 'SX', 'SI'}
                grp_to_save = '/'.join(grp_to_save + [utterance_id])
                print(grp_to_save, end=' ')
                if grp_to_save not in grp:
                    print('doing...')
                    # print(f)
                    abs_fn = os.path.join(subdir, f)
                    abs_fn_wav = os.path.join(subdir, f[:-3] + 'WAV')
                    feature = convert_one_feature(abs_fn_wav, tmp_mfc_file)
                    phoneme_target = get_phoneme_label(abs_fn)
                    grp_this = grp.create_group(grp_to_save)
                    grp_this.create_dataset('feature', data=feature)
                    grp_this.create_dataset('label', data=phoneme_target)
                    grp.file.flush()
                else:
                    print('done before!')

    os.remove(tmp_mfc_file)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_one_tf_record(h5obj, output_dir):
    name_split = h5obj.name.split('/')
    assert name_split[0] == ''
    name_split = name_split[1:]
    assert len(name_split) == 4
    sentenceid = name_split[-1]

    if sentenceid.startswith('SA'):
        return

    feature = h5obj['feature'][...]
    n_frame, n_feature = feature.shape
    assert n_feature == 26
    label = h5obj['label'][...]
    assert label.ndim == 1
    sentenceid_pure = int(sentenceid[2:])

    record_to_write = tf.train.Example(features=tf.train.Features(feature={
        'mfcc': tf.train.Feature(
            float_list=tf.train.FloatList(value=[feature.ravel().item(x) for x in range(len(feature.ravel()))])),
        'n_frame': _int64_feature(n_frame),
        'n_feature': _int64_feature(n_feature),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.item(x) for x in range(len(label))])),
        'sentence_id': _int64_feature(sentenceid_pure),
    }))

    # separate file for each instance. Thus, we can do filename level shuffling,
    # as at batch level, dynamic_pad is not available if we do shuffling.
    filename = os.path.join(output_dir, '_'.join(name_split) + '.tfrecords')
    print('writing {}'.format(filename))
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(record_to_write.SerializeToString())


def convert_hdf5_to_tf(infile, outfile_dir):
    # main function to convert hdf5 file to tensorflow standard record format.
    def write_to_tf_callback(name, obj):
        if isinstance(obj, h5py.Group) and 'feature' in obj:
            name_split = name.split('/')
            assert len(name_split) == 4
            write_one_tf_record(obj, outfile_dir)

    with h5py.File(infile, 'r') as infile_f:
        infile_f.visititems(write_to_tf_callback)


def compare_one_tf_record(h5obj, output_dir):
    name_split = h5obj.name.split('/')
    assert name_split[0] == ''
    name_split = name_split[1:]
    assert len(name_split) == 4
    sentenceid = name_split[-1]

    if sentenceid.startswith('SA'):
        return

    feature = h5obj['feature'][...]
    n_frame, n_feature = feature.shape
    assert n_feature == 26
    label = h5obj['label'][...]
    assert label.ndim == 1
    sentenceid_pure = int(sentenceid[2:])
    filename = os.path.join(output_dir, '_'.join(name_split) + '.tfrecords')
    print('verifying {}'.format(filename))
    # from <https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/2BsB4H97vM0>
    counter = 0
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        counter += 1
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        tf_mfcc = np.asarray(example.features.feature['mfcc'].float_list.value)
        tf_n_frame = example.features.feature['n_frame'].int64_list.value[0]
        tf_n_feature = example.features.feature['n_feature'].int64_list.value[0]
        tf_label = np.asarray(example.features.feature['label'].int64_list.value)
        tf_sentence_id = example.features.feature['sentence_id'].int64_list.value[0]
    assert counter == 1
    assert np.array_equal(tf_mfcc.reshape(tf_n_frame, tf_n_feature), feature)
    assert np.array_equal(tf_label, label)
    assert sentenceid_pure == tf_sentence_id


def compare_hdf5_and_tf(infile, outfile_dir):
    # main function to convert hdf5 file to tensorflow standard record format.
    def compare_with_tf_callback(name, obj):
        if isinstance(obj, h5py.Group) and 'feature' in obj:
            name_split = name.split('/')
            assert len(name_split) == 4
            compare_one_tf_record(obj, outfile_dir)

    with h5py.File(infile, 'r') as infile_f:
        infile_f.visititems(compare_with_tf_callback)


if __name__ == '__main__':
    timit_train_dir = os.path.join(dir_dictionary['datasets'], 'TIMITcorpus', 'TIMIT', 'TRAIN')
    timit_test_dir = os.path.join(dir_dictionary['datasets'], 'TIMITcorpus', 'TIMIT', 'TEST')
    trainset_file = os.path.join(dir_dictionary['features'], 'TIMIT_train.hdf5')
    testset_file = os.path.join(dir_dictionary['features'], 'TIMIT_test.hdf5')

    with h5py.File(trainset_file) as f:
        main_loop(timit_train_dir, f)
    with h5py.File(testset_file) as f:
        main_loop(timit_test_dir, f)

    # make tf compatible data set
    trainset_tf = os.path.join(dir_dictionary['features'], 'TIMIT_train_tf')
    testset_tf = os.path.join(dir_dictionary['features'], 'TIMIT_test_tf')
    if not os.path.exists(trainset_tf):
        os.makedirs(trainset_tf)

    if not os.path.exists(testset_tf):
        os.makedirs(testset_tf)

    convert_hdf5_to_tf(trainset_file, trainset_tf)
    compare_hdf5_and_tf(trainset_file, trainset_tf)
    convert_hdf5_to_tf(testset_file, testset_tf)
    compare_hdf5_and_tf(testset_file, testset_tf)


    # test that hdf5 is preserved

    # a debugging part to for later siamese mapping.

    # sentence_dict = {
    #     'SA': set(),
    #     'SX': set(),
    #     'SI': set(),
    # }
    #
    #
    # def check_number_call_back(name, obj):
    #     if isinstance(obj, h5py.Group) and 'feature' in obj:
    #         name_split = name.split('/')
    #         assert len(name_split) == 4
    #         sentenceid = name_split[-1]
    #         sentencetype = sentenceid[:2]
    #         sentenceid_pure = sentenceid[2:]
    #         sentence_dict[sentencetype] = sentence_dict[sentencetype] | {int(sentenceid_pure)}
    #
    #
    # # let's scan the whole set of sentence numbers.
    # with h5py.File(trainset_file, 'r') as f:
    #     f.visititems(check_number_call_back)
    # # you should add test set to make the assertions hold.
    # with h5py.File(testset_file, 'r') as f:
    #     f.visititems(check_number_call_back)
    # assert sentence_dict['SA'] == set(range(1, 3))
    # assert sentence_dict['SX'] == set(range(3, 453))
    # assert sentence_dict['SI'] == set(range(453, 2343))
    #
    # # this is a debugging part, where I just load all the train data, normalize them,
    # # and then write them out as npy files, for use with the initial version of
    # # <https://github.com/dresen/tensorflow_CTC_example>
    #
    # trainset_numpy_feature = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'feature')
    # trainset_numpy_label = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'label')
    # trainset_numpy_sentence = os.path.join(dir_dictionary['features'], 'TIMIT_train', 'sentence_label')
    # if not os.path.exists(trainset_numpy_feature):
    #     os.makedirs(trainset_numpy_feature)
    #
    # if not os.path.exists(trainset_numpy_label):
    #     os.makedirs(trainset_numpy_label)
    #
    # if not os.path.exists(trainset_numpy_sentence):
    #     os.makedirs(trainset_numpy_sentence)
    #
    # feature_list = []
    # label_list = []
    # sentence_list = []
    # name_list = []
    #
    #
    # # then load all the train data
    # def train_data_call_back(name, obj):
    #     if isinstance(obj, h5py.Group) and 'feature' in obj:
    #         name_split = name.split('/')
    #         assert len(name_split) == 4
    #         if name_split[-1].startswith('SA'):
    #             return
    #         else:
    #             feature = obj['feature'][...]
    #             label = obj['label'][...]
    #             feature_list.append(feature)
    #             label_list.append(label)
    #             name_list.append('_'.join(name_split))
    #
    #             # add sentence id.
    #             sentenceid = name_split[-1]
    #             sentencetype = sentenceid[:2]
    #             sentenceid_pure = int(sentenceid[2:])
    #             sentence_list.append(sentenceid_pure)
    #
    #
    # with h5py.File(trainset_file, 'r') as f:
    #     f.visititems(train_data_call_back)
    # assert len(feature_list) == len(label_list) == len(name_list) == 3696  # excluding the SA sentences.
    # assert len(set(name_list)) == len(name_list)
    # feature_all = np.concatenate(feature_list, axis=0)
    # print(feature_all.shape)
    # # then normalize the data
    # mean_all = feature_all.mean(axis=0, keepdims=True)
    # std_all = feature_all.std(axis=0, keepdims=True)
    # assert mean_all.shape == std_all.shape == (1, 26)
    #
    # # then write them out in numpy files.
    # for name, feature, label, sentence_label in zip(name_list, feature_list, label_list, sentence_list):
    #     feature_this = (feature - mean_all) / std_all
    #     assert label.ndim == 1
    #     np.save(os.path.join(trainset_numpy_feature, name + '.npy'), feature_this.T)
    #     np.save(os.path.join(trainset_numpy_label, name + '.npy'), label)
    #     np.save(os.path.join(trainset_numpy_sentence, name + '.npy'), np.uint16(sentence_label))
    #
    #     # # save sentence id.
    #     # sentence_list_np = np.array(sentence_list, dtype=np.uint32)
    #     # assert np.array_equal(sentence_list_np, np.array(sentence_list))
    #     # assert sentence_list_np.shape == (3696,)
    #     # np.save(trainset_numpy_sentence_file, sentence_list_np)
