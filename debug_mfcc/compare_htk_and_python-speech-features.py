from __future__ import absolute_import, print_function, division, unicode_literals
from htkmfc import HTKFeat_read
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import scale
import matplotlib
matplotlib.use('Agg')  # for doing headless
import matplotlib.pyplot as plt

def get_HTK_feature():
    read_obj = HTKFeat_read()
    read_obj.open('./SX442.mfc')
    feature = read_obj.getall()
    read_obj.fh.close()
    return feature

# my main conclusion about HTK vs python_speech_features is that they are roughly the same for those 12 coefficients,
# but very different for 0th, and similar for energy.
# HTK has ability to do Hamming, and it uses log of amplitude, instead of power by default.
# In actual feature generation, I stick to default of HTK, and 5.1 of Graves' book as much as possible.
# compare mfcc_config and mfcc_config_debug.

def get_python_speech_features_feature():
    (rate, sig) = wav.read('./SX442_sox.wav')
    assert rate == 16000
    mfcc_feat = mfcc(sig, samplerate=rate, appendEnergy=True)
    mfcc_feat = np.concatenate((mfcc_feat[:,1:], mfcc_feat[:,:1]), axis=1)
    return mfcc_feat

if __name__ == '__main__':
    HTK_feature = get_HTK_feature()
    PSF_feature = get_python_speech_features_feature()
    print(HTK_feature.shape, PSF_feature.shape)
    print(HTK_feature.dtype, PSF_feature.dtype)
    test_len = 358
    assert HTK_feature.shape[0] >= test_len
    assert PSF_feature.shape[0] >= test_len
    HTK_feature = HTK_feature[:test_len]
    PSF_feature = PSF_feature[:test_len]
    assert HTK_feature.shape[1] == PSF_feature.shape[1] == 13
    assert HTK_feature.ndim == PSF_feature.ndim == 2

    # ok, first compare the first 12 coefficients
    raw_12_HTK = HTK_feature[:,:12]
    raw_12_PSF = PSF_feature[:,:12]
    print('correlation between first 12 coefficients', pearsonr(raw_12_HTK.ravel(), raw_12_PSF.ravel()))
    print(raw_12_HTK[:5] - raw_12_PSF[:5])

    raw_0_HTK = HTK_feature[:, 12:]
    raw_0_PSF = PSF_feature[:, 12:]
    print('correlation between 0th coefficients', pearsonr(raw_0_HTK.ravel(), raw_0_PSF.ravel()))
    print(raw_0_HTK[:5]-raw_0_PSF[:5])

    print('correlation between raw coefficients', pearsonr(HTK_feature.ravel(), PSF_feature.ravel()))
    # normalize them
    HTK_feature_norm = scale(HTK_feature.astype(np.float64))
    PSF_feature_norm = scale(PSF_feature.astype(np.float64))
    print('correlation between normalized coefficients', pearsonr(HTK_feature_norm.ravel(), PSF_feature_norm.ravel()))
    plt.scatter(HTK_feature_norm.ravel(), PSF_feature_norm.ravel())
    plt.xlabel('HTK normalized features')
    plt.ylabel('PSF normalized features')
    plt.savefig('PSF_vs_HTK.png', dpi=300)

