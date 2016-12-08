from __future__ import absolute_import, division, print_function
import numpy as np
from itertools import combinations


def load_batched_data_sentence(sentence_names_all, sentence_id_all, seed=0, neg_pair_for_each_positive_pair=2):
    assert len(sentence_id_all) == len(sentence_names_all) == 1344
    rng_state = np.random.RandomState(seed=seed)
    # ok. let's create
    unique_sentence_ids, sentence_id_counts = np.unique(sentence_id_all, return_counts=True)
    loc_all = np.arange(len(sentence_names_all))

    positive_sentence_mask = sentence_id_counts > 1
    print('unique id count {},'
          'positive set {}'.format(unique_sentence_ids.size, positive_sentence_mask.sum()))
    assert (sentence_id_counts[positive_sentence_mask] == 7).all() and positive_sentence_mask.sum() == 120

    positive_unique_sentence_ids = unique_sentence_ids[positive_sentence_mask]

    # I will iterate all positive pairs. For each positive pair, I will create (roughly) equal number of negative pairs
    # for each of two, with sum of negative pairs equal neg_pair_for_each_positive_pair.
    assert neg_pair_for_each_positive_pair > 1
    neg_pair_for_l = neg_pair_for_each_positive_pair // 2
    neg_pair_for_r = neg_pair_for_each_positive_pair - neg_pair_for_l

    signal_list_all = []
    pair_all = []

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
            pair_name_list = [(sentence_names_all[l], sentence_names_all[r]) for (l, r) in pair_list]
            # print(pair_list)
            # print(pair_name_list)
            for p in pair_name_list[:1]:
                assert p[0].split('/')[-1] == p[1].split('/')[-1]
            for p in pair_name_list[1:]:
                assert p[0].split('/')[-1] != p[1].split('/')[-1]
            signal_list_this = np.concatenate((np.ones(1, dtype=np.bool_),
                                               np.zeros(neg_pair_for_each_positive_pair, dtype=np.bool_)))
            assert signal_list_this.shape == (len(pair_list),) == (len(pair_name_list),)

            signal_list_all.append(signal_list_this)
            pair_all.extend(pair_name_list)

    signal_list_all = np.concatenate(signal_list_all)
    pair_total = 120 * 7 * 6 // 2 * (1 + neg_pair_for_each_positive_pair)
    assert signal_list_all.shape == (pair_total,)
    assert len(pair_all) == pair_total

    return pair_all, signal_list_all
