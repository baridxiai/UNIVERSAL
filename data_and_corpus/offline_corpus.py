# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from logging import getLogger
import math
import numpy as np


logger = getLogger()

def offline(data_path, mono=False):
    print("offline mode")
    if mono:
        dataset_path = []
        for lang in data_path:
            dataset_path.append(tf.data.TextLineDataset(lang))
        train_examples = tf.data.Dataset.zip(tuple(dataset_path))
    else:
        src, tgt = data_path
        dataset_0 = tf.data.TextLineDataset(src)
        dataset_1 = tf.data.TextLineDataset(tgt)
        train_examples = tf.data.Dataset.zip((dataset_0, dataset_1))
    return train_examples

def offline_multi(data_path):
    print("offline mode")
    dataset_path = []
    dataset_ratio = []
    for lang in data_path:
        dataset_path.append(tf.data.TextLineDataset(lang[0]))
        dataset_ratio.append(lang[1])
    train_examples = tf.data.Dataset.sample_from_datasets(
    dataset_path, weights=dataset_ratio, stop_on_empty_dataset=True)
    return train_examples


class Dataset(object):

    def __init__(self, sent, pos, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # # remove empty sentences
        # self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos) == (self.sent[self.pos[:, 1]] == eos).sum()  # check sentences indices
        # assert self.lengths.min() > 0                                     # check empty sentences

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos = self.pos[a:b]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # re-index
        min_pos = self.pos.min()
        max_pos = self.pos.max()
        self.pos -= min_pos
        self.sent = self.sent[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.pos[sentence_ids]
            sent = [self.sent[a:b] for a, b in pos]
            sent = self.batch_sentences(sent)
            yield (sent, sentence_ids) if return_indices else sent

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, seed=None, return_indices=False):
        """
        Return a sentences iterator.
        """
        assert seed is None or shuffle is True and type(seed) is int
        rng = np.random.RandomState(seed)
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        # sentence lengths
        lengths = self.lengths + 2

        # select sentences to iterate over
        if shuffle:
            indices = rng.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            rng.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)


class ParallelDataset(Dataset):

    def __init__(self, sent1, pos1, sent2, pos2, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == len(self.pos2) > 0                          # check number of sentences
        assert len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()  # check sentences indices
        assert len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()  # check sentences indices
        assert eos <= self.sent1.min() < self.sent1.max()                    # check dictionary indices
        assert eos <= self.sent2.min() < self.sent2.max()                    # check dictionary indices
        assert self.lengths1.min() > 0                                       # check empty sentences
        assert self.lengths2.min() > 0                                       # check empty sentences

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos1)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos1 = self.pos1[a:b]
        self.pos2 = self.pos2[a:b]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # re-index
        min_pos1 = self.pos1.min()
        max_pos1 = self.pos1.max()
        min_pos2 = self.pos2.min()
        max_pos2 = self.pos2.max()
        self.pos1 -= min_pos1
        self.pos2 -= min_pos2
        self.sent1 = self.sent1[min_pos1:max_pos1 + 1]
        self.sent2 = self.sent2[min_pos2:max_pos2 + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
            yield (sent1, sent2, sentence_ids) if return_indices else (sent1, sent2)

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1, return_indices=False):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths1 + self.lengths2 + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind='mergesort')]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum([lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)