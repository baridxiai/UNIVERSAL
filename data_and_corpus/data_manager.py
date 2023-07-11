# -*- coding: utf-8 -*-
# code warrior: Barid

# import sentencepiece as sp
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import pre_tokenizers, decoders, trainers, processors
import tensorflow as tf
import fastBPE

# import fasttext
# lang_checker = fasttext.load_model('../lid.176.ftz')


class DatasetManager(object):
    def __init__(
        self,
        tokenizer_path,
        training_set,
        cut_length=None,
        tokenization=False,
        dev_set=None,
        test_set=None,
        domain_index=[],
        mono=False,
        lang_dict = {
    "en":1,
}
    ):
        # self.tokenizer = Tokenizer(BPE(continuing_subword_prefix="@@"))
        # self.tokenizer.decoder = decoders.ByteLevel()
        # self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        # self.tokenizer.pre_tokenizer = Whitespace()
        # self.tokenizer.model = BPE.from_file(
        #     tokenizer_path + 'vocab.json', tokenizer_path + 'merges.txt', unk_token="[UNK]")
        vocab = WordLevel.read_file(tokenizer_path[0])
        bpe_tok = WordLevel(vocab, unk_token="[UNK]")
        self.tokenizer = Tokenizer(bpe_tok)
        self.bpe_tok = fastBPE.fastBPE(tokenizer_path[1])
        self.domain_index = domain_index
        self.train_examples = training_set
        self.test_examples = test_set
        self.dev_examples = dev_set
        self.cut_length = cut_length
        self.tokenization = tokenization
        self.mono = mono
        self.lang_dict =  lang_dict
        self.tokenizer.add_special_tokens(["[EOS]", "[SOS]", "[PAD]", "[MAK]", "[UNK]"])

    def get_domain_index(self):
        return self.domain_index

    def set_domain_index(self, domain_index):
        self.domain_index = domain_index

    def tokenize_encoding(self, lang1, lang2):
        if self.tokenization:
            lang1 = self.encode(lang1.numpy().decode())
            lang2 = self.encode(lang2.numpy().decode())
        else:
            lang1 = list(map(int, lang1.numpy().decode().split()))
            lang2 = list(map(int, lang2.numpy().decode().split()))
        if self.cut_length is not None:
            lang1 = lang1[: self.cut_length]
            lang2 = lang2[: self.cut_length]
        return lang1, lang2

    def tf_tokenize(self, lang1, lang2):
        if self.tokenization:
            lang1, lang2 = tf.py_function(
                self.tokenize_encoding, [lang1, lang2], [tf.int32, tf.int32]
            )
        else:
            lang1 = tf.cast(tf.strings.to_number(tf.strings.split(lang1)), tf.int32)
            lang2 = tf.cast(tf.strings.to_number(tf.strings.split(lang2)), tf.int32)
        lang1.set_shape([None])
        lang2.set_shape([None])
        return lang1, lang2

    def tf_tokenize_mono(self, *args):
        langs_mono = []
        for lang in args:
            lang_re = tf.cast(tf.strings.to_number(tf.strings.split(lang)), tf.int32)
            lang_re.set_shape([None])
            langs_mono.append(lang_re)
        return langs_mono

    def encode(self, string):
        string = self.bpe_tok.apply([string])[0]
        return self.tokenizer.encode(string.strip().split(" "), is_pretokenized=True).ids

    def decode(self, string):
        return self.tokenizer.decode(string)

    def get_vocabulary_size(self):
        # return self.tokenizer.vocab_size
        return len(self.tokenizer.get_vocab())

    def get_raw_train_dataset(self, shuffle=40):
        self.train_examples = self.train_examples.shuffle(shuffle)
        if self.mono:
            return self.train_examples.map(
                self.tf_tokenize_mono, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return self.train_examples.map(
            self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def get_raw_dev_dataset(self):
        if self.mono:
            return self.dev_examples.map(
                self.tf_tokenize_mono, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return self.dev_examples.map(
            self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    def get_raw_test_dataset(self):
        return self.test_examples.map(self.tf_tokenise)

class DatasetManager_multi(DatasetManager):

    def tf_mono_preprocess(self, inputs):
        inputs = tf.strings.split(inputs,"@@")
        inputs = tf.cast(tf.strings.to_number(tf.strings.split(inputs)), tf.int32)
        return inputs
    def get_raw_train_dataset(self, shuffle=40):
        self.train_examples = self.train_examples.shuffle(shuffle)
        if self.mono:
        #     return self.train_examples.map(
        #         lambda x: tf.numpy_function(func=self.tf_mono_preprocess,
        #   inp=[x], Tout=[tf.int32,tf.int32]), num_parallel_calls=tf.data.experimental.AUTOTUNE
        #     )
            return self.train_examples.map(
                self.tf_mono_preprocess
          , num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return self.train_examples.map(
            self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

    def get_raw_dev_dataset(self):
        if self.mono:
            return self.dev_examples.map(
                self.tf_mono_preprocess
          , num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return self.dev_examples.map(
            self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )