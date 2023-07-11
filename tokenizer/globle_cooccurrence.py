from __future__ import division
from collections import Counter, defaultdict
import os
import json

# from functools import map, reduce
import json
from ast import literal_eval
from os.path import exists
import numpy as np
from numpy import save

NULL_WORD = "<null>"


def percent(current, total):
    import sys

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = "#" * int(current / total * bar_length)
    spaces = " " * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total)))


class Corpus(object):
    def __init__(self, **kwargs):
        """
        `size`: How many words on each side to select for each window. Specifying
        `size` gives symmetric context windows and is equivalent to setting
        `left_size` and `right_size` to the same value.
        """
        if "size" in kwargs:
            self.left_size = self.right_size = kwargs["size"]
        elif "left_size" in kwargs or "right_size" in kwargs:
            self.left_size = kwargs.get("left_size", 0)
            self.right_size = kwargs.get("right_size", 0)
        else:
            raise KeyError("At least one of `size`, `left_size`, and `right_size` must be given")
        self._words = None
        self._word_index = None
        s_path = "../UNIVERSAL/vocabulary/DeEn_6k_wiki/cooccurrence_matrix"
        if exists(s_path):
            obj = json.loads(
                open("../UNIVERSAL/vocabulary/DeEn_6k_wiki/cooccurrence_matrix").read()
            )
            obj = json.loads(obj)
            self._cooccurrence_matrix = {literal_eval(k): v for k, v in obj.items()}
        else:
            self._cooccurrence_matrix = None
        self.constant_counter = 1

    def tokenized_regions(self):
        """
        Returns an iterable of all tokenized regions of text from the corpus.
        """
        return map(self.tokenize, self.extract_regions())
        # pass

    def fit(self, vocab_size=None, min_occurrences=1):
        self._cooccurrence_matrix = np.zeros([vocab_size, vocab_size], dtype=np.float32)
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        num_n = 0
        for region in self.tokenized_regions():
            num_n += 1
            if num_n % 10000 == 0:
                percent(num_n, 28112811 + 73800738)
            word_counts.update(region)
            for left_context, word, right_context in self.region_context_windows(region):
                # self._cooccurrence_matrix[int(word), int(word)] += 1
                for i, context_word in enumerate(left_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    self._cooccurrence_matrix[
                        int(word), int(context_word)
                    ] += self.constant_counter / (i + 1)
                for i, context_word in enumerate(right_context):
                    self._cooccurrence_matrix[
                        int(word), int(context_word)
                    ] += self.constant_counter / (i + 1)
            if num_n % 10000000 == 0:
                save("globle_cooccurrence.npy", self._cooccurrence_matrix)
        save("data_n3.npy", self._cooccurrence_matrix)

    def is_fit(self):
        """
        Returns a boolean for whether or not the Corpus object has been fit to
        the text yet.
        """
        return self._words is not None

    def extract_regions(self):
        """
        Returns an iterable of all regions of text (strings) in the corpus that
        should each be considered one contiguous unit. Messages, comments,
        reviews, etc.
        """
        raise NotImplementedError()

    @staticmethod
    def tokenize(string):
        """
        Takes strings that come from `extract_regions` and returns the tokens
        from that string.
        """
        raise NotImplementedError()

    @staticmethod
    def window(region, start_index, end_index):
        """
        Returns the list of words starting from `start_index`, going to `end_index`
        taken from region. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in region, this function will pad
        its return value with `NULL_WORD`.
        """
        last_index = len(region) + 1
        selected_tokens = region[max(start_index, 0) : min(end_index, last_index) + 1]
        return selected_tokens

    def region_context_windows(self, region):
        for i, word in enumerate(region):
            start_index = i - self.left_size
            end_index = i + self.right_size
            left_context = self.window(region, start_index, i - 1)
            right_context = self.window(region, i + 1, end_index)
            yield (left_context, word, right_context)

    @property
    def words(self):
        if not self.is_fit():
            self.fit()
        return self._words

    @property
    def word_index(self):
        if not self.is_fit():
            self.fit()
        return self._word_index

    @property
    def cooccurrence_matrix(self):
        if self._cooccurrence_matrix is None:
            self.fit()
        return self._cooccurrence_matrix


class CoocMatr(Corpus):
    def __init__(self, path, **kwargs):
        """
        If `path` is a file, it will be treated as the only file in the corpus.
        If it's a directory, every file not starting with a dot (".") will be
        considered to be in the corpus. See the constructor for `Corpus` for
        keyword args.
        """
        super(CoocMatr, self).__init__(**kwargs)
        self.file_paths = path

    def extract_regions(self):
        # This is not exactly a rock-solid way to get the body, but it's ~2x as
        # fast as json parsing each line
        # import fileinput

        for k, file_path in enumerate(self.file_paths):
            self.constant_counter = file_path[1]
            with open(file_path[0]) as file_:
                for k, line in enumerate(file_):
                    line_list = line.strip().split(" 2 ")
                    for _, line in enumerate(line_list):
                        yield line

        # for line in fileinput.input(self.file_paths):
        #     yield line

    @staticmethod
    def tokenize(string):
        return string.strip().split(" ")


files = [
    ("/home/vivalavida/massive_data/data/wiki/train.de.60000.withVoc.BPE-256-STREAM-DeEn6k_vocab", 2.5),
    ("/home/vivalavida/massive_data/data/wiki/train.en.60000.withVoc.BPE-256-STREAM-DeEn6k_vocab", 1),
]

tool = CoocMatr(files, size=3)
tool.fit(vocab_size=72199)
