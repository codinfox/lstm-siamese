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


if __name__ == '__main__':
    timit_train_dir = os.path.join(dir_dictionary['datasets'], 'TIMITcorpus', 'TIMIT', 'TRAIN')
    timit_test_dir = os.path.join(dir_dictionary['datasets'], 'TIMITcorpus', 'TIMIT', 'TEST')
    trainset_file = os.path.join(dir_dictionary['features'], 'TIMIT_train.hdf5')
    testset_file = os.path.join(dir_dictionary['features'], 'TIMIT_test.hdf5')

    with h5py.File(trainset_file) as f:
        main_loop(timit_train_dir, f)
    with h5py.File(testset_file) as f:
        main_loop(timit_test_dir, f)
