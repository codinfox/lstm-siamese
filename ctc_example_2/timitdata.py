#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: timitdata.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from tensorpack import RNGDataFlow, ProxyDataFlow
import os, sys, glob
import numpy as np
import string
from six.moves import range

CHARSET = set(string.ascii_lowercase + ' ')
def read_timit_txt(f):
    f = open(f)
    line = f.readlines()[0].strip().split(' ')
    line = line[2:]
    line = ' '.join(line)
    line = line.replace('.', '').lower()
    line = filter(lambda c: c in CHARSET, line)
    f.close()
    return line

def label_sentence(sent):
    ret = []
    for c in sent:
        if c == ' ':
            ret.append(26)
        else:
            ret.append(ord(c) - ord('a'))
    return np.asarray(ret)

class TIMITData(RNGDataFlow):
    def __init__(self):
        filelists = open('../TRAIN/filelist.txt').readlines()
        files = [os.path.join('../TRAIN', p.strip()) for p in filelists]
        self.featfiles = [p[:-4] + '.npy' for p in files]
        sent = [read_timit_txt(p[:-4] + '.TXT') for p in files]
        self.labels = [label_sentence(s) for s in sent]

    def size(self):
        return len(self.featfiles)

    def get_data(self):
        idxs = list(range(self.size()))
        self.rng.shuffle(idxs)
        for t in idxs:
            feat = np.load(self.featfiles[t])   #len x 39
            lab = self.labels[t]    # w
            yield [feat, lab]

def batch_feature(feats):
    maxlen = max([k.shape[0] for k in feats])
    bsize = len(feats)
    ret = np.zeros((bsize, maxlen, feats[0].shape[1]))
    for idx, feat in enumerate(feats):
        ret[idx,:feat.shape[0],:] = feat
    return ret

def sparse_label(labels):
    maxlen = max([k.shape[0] for k in labels])
    shape = [len(labels), maxlen]   # bxt
    indices = []
    values = []
    for bid, lab in enumerate(labels):
        for tid, c in enumerate(lab):
            indices.append([bid, tid])
            values.append(c)
    indices = np.asarray(indices)
    values = np.asarray(values)
    return (indices, values, shape)

class TIMITBatch(ProxyDataFlow):
    def __init__(self, ds, batch):
        self.batch = batch
        self.ds = ds

    def size(self):
        return self.ds.size() // self.batch

    def get_data(self):
        itr = self.ds.get_data()
        for _ in range(self.size()):
            feats = []
            labs = []
            for b in range(self.batch):
                feat, lab = next(itr)
                feats.append(feat)
                labs.append(lab)
            batchfeat = batch_feature(feats)
            batchlab = sparse_label(labs)
            seqlen = np.asarray([k.shape[0] for k in feats])
            yield [batchfeat, batchlab, seqlen]

if __name__ == '__main__':
    d = TIMITData()
    d = TIMITBatch(d, 10)
    d.reset_state()
    for k in d.get_data():
        break
