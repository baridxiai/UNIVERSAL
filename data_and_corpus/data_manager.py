# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
# import sentencepiece as sp
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
# import fasttext
# lang_checker = fasttext.load_model('../lid.176.ftz')


class DatasetManager(object):
    def __init__(self, tokenizer_path, training_set, dev_set=None, test_set=None, domain_index=[]):
        self.tokenizer = Tokenizer(BPE(continuing_subword_prefix="@@"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.model = BPE.from_file(
            tokenizer_path + 'vocab.json', tokenizer_path + 'merges.txt', unk_token="[UNK]",)
        self.domain_index = domain_index
        self.train_examples = training_set
        self.test_examples = test_set
        self.dev_examples = dev_set

        self.tokenizer.add_special_tokens(
            ["[EOS]", "[SOS]", "[PAD]", "[MAK]", "[UNK]"])

    def get_domain_index(self):
        return self.domain_index

    def set_domain_index(self, domain_index):
        self.domain_index = domain_index

    def tokenize_encoding(self, lang1, lang2):
        lang1 = self.tokenizer.encode(lang1.numpy().decode()).ids
        lang2 = self.tokenizer.encode(lang2.numpy().decode()).ids
        return lang1, lang2

    def tf_tokenize(self, lang1, lang2):
        lang1, lang2 = tf.py_function(self.tokenize_encoding, [lang1, lang2],
                                      [tf.int32, tf.int32])
        lang1.set_shape([None])
        lang2.set_shape([None])
        return lang1, lang2

    def encode(self, string):
        return self.tokenizer.encode(string).ids

    def decode(self, string):
        return self.tokenizer.decode(string)

    def get_vocabulary_size(self):
        # return self.tokenizer.vocab_size
        return len(self.tokenizer.get_vocab())

    def get_raw_train_dataset(self):
        return self.train_examples.map(
            self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_raw_dev_dataset(self):
        return self.dev_examples.map(self.tf_tokenise)

    def get_raw_test_dataset(self):
        return self.test_examples.map(self.tf_tokenise)
