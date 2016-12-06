import numpy as np
import os
import os.path
from itertools import combinations


def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]
    return (np.array(indices), np.array(vals), np.array(shape))


def data_lists_to_batches(inputList, targetList, batchSize):
    '''Takes a list of input matrices and a list of target arrays and returns
       a list of batches, with each batch being a 3-element tuple of inputs,
       targets, and sequence lengths.
       inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
       targetList: list of 1-d arrays or lists of ints
       batchSize: int indicating number of inputs/targets per batch
       returns: dataBatches: list of batch data tuples, where each batch tuple (inputs, targets, seqLengths) consists of
                    inputs = 3-d array w/ shape nTimeSteps x batchSize x nFeatures
                    targets = tuple required as input for SparseTensor
                    seqLengths = 1-d array with int number of timesteps for each sample in batch
                maxSteps: maximum number of time steps across all samples'''

    assert len(inputList) == len(targetList)
    nFeatures = inputList[0].shape[0]
    maxSteps = 0
    for inp in inputList:
        maxSteps = max(maxSteps, inp.shape[1])

    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((maxSteps, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            padSecs = maxSteps - inputList[origI].shape[1]
            batchInputs[:, batchI, :] = np.pad(inputList[origI].T, ((0, padSecs), (0, 0)),
                                               'constant', constant_values=0)
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, target_list_to_sparse_tensor(batchTargetList),
                            batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxSteps)


def load_batched_data(specPath, targetPath, batchSize):
    '''returns 3-element tuple: batched data (list), max # of time steps (int), and
       total number of samples (int)'''
    dir_list_1 = sorted(os.listdir(specPath))
    dir_list_2 = sorted(os.listdir(targetPath))
    assert dir_list_1 == dir_list_2
    for x in dir_list_1:
        assert x.endswith('.npy')

    return data_lists_to_batches([np.load(os.path.join(specPath, fn)) for fn in dir_list_1],
                                 [np.load(os.path.join(targetPath, fn)) for fn in dir_list_2],
                                 batchSize) + \
           (len(dir_list_1),)


def load_batched_data_sentence(featurePath, sentencePath, seed=0, neg_pair_for_each_positive_pair=2):
    dir_list_1 = sorted(os.listdir(featurePath))
    dir_list_2 = sorted(os.listdir(sentencePath))
    assert dir_list_1 == dir_list_2

    for x in dir_list_1:
        assert x.endswith('.npy')

    sentence_id_list = [
        x.split('/')[-1].split('_')[-1][:-4] for x in dir_list_1
        ]
    # print(sentence_id_list)

    feature_all = [np.load(os.path.join(featurePath, fn)) for fn in dir_list_1]
    sentence_id_all = np.array([np.load(os.path.join(sentencePath, fn)) for fn in dir_list_1])
    assert sentence_id_all.shape == (len(feature_all),)
    loc_all = np.arange(len(feature_all))

    rng_state = np.random.RandomState(seed=seed)

    # ok. let's create
    unique_sentence_ids, sentence_id_counts = np.unique(sentence_id_all, return_counts=True)

    positive_sentence_mask = sentence_id_counts > 1
    print('unique id count {},'
          'positive set {}'.format(unique_sentence_ids.size, positive_sentence_mask.sum()))
    assert (sentence_id_counts[positive_sentence_mask] == 7).all() and positive_sentence_mask.sum() == 330

    positive_unique_sentence_ids = unique_sentence_ids[positive_sentence_mask]

    # I will iterate all positive pairs. For each positive pair, I will create (roughly) equal number of negative pairs
    # for each of two, with sum of negative pairs equal neg_pair_for_each_positive_pair.
    assert neg_pair_for_each_positive_pair > 1
    neg_pair_for_l = neg_pair_for_each_positive_pair // 2
    neg_pair_for_r = neg_pair_for_each_positive_pair - neg_pair_for_l

    signal_list_all = []
    feature_list_left_all = []
    feature_list_right_all = []

    for pos_id in positive_unique_sentence_ids:
        # first, find all indices equal to this.
        positive_loc_this = loc_all[sentence_id_all == pos_id]
        negative_loc_this = loc_all[sentence_id_all != pos_id]

        # then, do a combination of all pairs in postive_loc_this.
        for pos_pair_l, pos_pair_r in combinations(positive_loc_this, 2):
            # then choose neg_pair_for_l for left

            neg_pair_list = rng_state.choice(negative_loc_this, size=neg_pair_for_each_positive_pair,
                                             replace=True)
            neg_pair_list_l = neg_pair_list[:neg_pair_for_l]
            neg_pair_list_r = neg_pair_list[neg_pair_for_l:]
            assert neg_pair_list_l.shape == (neg_pair_for_l,)
            assert neg_pair_list_r.shape == (neg_pair_for_r,)

            # first emit this positive pair
            pair_list = [(pos_pair_l, pos_pair_r)] + [(pos_pair_l, x) for x in neg_pair_list_l] + [(pos_pair_r, x) for x
                                                                                                   in neg_pair_list_r]
            pair_name_list = [(sentence_id_list[l], sentence_id_list[r]) for (l, r) in pair_list]
            # print(pair_list)
            # print(pair_name_list)
            for p in pair_list:
                assert p[0] != p[1]
            for p in pair_name_list[:1]:
                assert p[0] == p[1]
            for p in pair_name_list[1:]:
                assert p[0] != p[1]
            signal_list_this = np.concatenate((np.ones(1, dtype=np.bool_),
                                               np.zeros(neg_pair_for_each_positive_pair, dtype=np.bool_)))
            assert signal_list_this.shape == (len(pair_list),) == (len(pair_name_list),)

            signal_list_all.append(signal_list_this)
            feature_list_left_all.extend([feature_all[l] for (l, _) in pair_list])
            feature_list_right_all.extend([feature_all[r] for (_, r) in pair_list])

    signal_list_all = np.concatenate(signal_list_all)
    pos_pair_total = 330 * 7 * 6 // 2 * (1 + neg_pair_for_each_positive_pair)
    assert signal_list_all.shape == (pos_pair_total,)
    assert len(feature_list_left_all) == len(feature_list_right_all) == pos_pair_total

    return feature_list_left_all, feature_list_right_all, signal_list_all
