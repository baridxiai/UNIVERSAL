# -*- coding: utf-8 -*-
# code warrior: Barid

from UNIVERSAL.data_and_corpus import dataset_preprocessing
import tensorflow as tf

# import fasttext
# lang_checker = fasttext.load_model('../lid.176.ftz')


class DatasetManager(object):
    def __init__(
        self,
        vocab,
        bpe,
        training_set,
        parameters,
        preprocess_fn=None,
        postprocess_fn=None,
        dev_set=None,
    ):
        self.tokenizer = vocab
        self.bpe_tok = bpe
        self.train_examples = training_set
        self.dev_examples = dev_set
        self.cut_length = parameters["max_sequence_length"]
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.parameters = parameters
        self.data_opt = tf.data.Options()
        self.data_opt.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )
    # def tokenize_encoding(self, lang1, lang2):
    #     if self.tokenization:
    #         lang1 = self.encode(lang1.numpy().decode())
    #         lang2 = self.encode(lang2.numpy().decode())
    #     else:
    #         lang1 = list(map(int, lang1.numpy().decode().split()))
    #         lang2 = list(map(int, lang2.numpy().decode().split()))
    #     if self.cut_length is not None:
    #         lang1 = lang1[: self.cut_length]
    #         lang2 = lang2[: self.cut_length]
    #     return lang1, lang2

    # def tf_tokenize(self, lang1, lang2):
    #     if self.tokenization:
    #         lang1, lang2 = tf.py_function(
    #             self.tokenize_encoding, [lang1, lang2], [tf.int32, tf.int32]
    #         )
    #     else:
    #         lang1 = tf.cast(tf.strings.to_number(tf.strings.split(lang1)), tf.int32)
    #         lang2 = tf.cast(tf.strings.to_number(tf.strings.split(lang2)), tf.int32)
    #     lang1.set_shape([None])
    #     lang2.set_shape([None])
    #     return lang1, lang2

    # def tf_tokenize_mono(self, *args):
    #     langs_mono = []
    #     for lang in args:
    #         lang_re = tf.cast(tf.strings.to_number(tf.strings.split(lang)), tf.int32)
    #         lang_re.set_shape([None])
    #         langs_mono.append(lang_re)
    #     return langs_mono
    # def get_raw_train_dataset(self, shuffle=40):
    #     self.train_examples = self.train_examples.shuffle(shuffle)
    #     if self.mono:
    #         return self.train_examples.map(
    #             self.tf_tokenize_mono, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #         )
    #     return self.train_examples.map(
    #         self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )

    # def get_raw_dev_dataset(self):
    #     if self.mono:
    #         return self.dev_examples.map(
    #             self.tf_tokenize_mono, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #         )
    #     return self.dev_examples.map(
    #         self.tf_tokenize, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )

    # def get_raw_test_dataset(self):
    #     return self.test_examples.map(self.tf_tokenise)
    def encode(self, string, bpe_only=False):
        string = self.bpe_tok.apply([string])[0]
        if bpe_only:
            return string
        return self.tokenizer.encode(string.strip().split(" "), is_pretokenized=True).ids

    def decode(self, string):
        return self.tokenizer.decode(string)

    def get_vocabulary_size(self):
        # return self.tokenizer.vocab_size
        return len(self.tokenizer.get_vocab())
    def template(self, inputs):
        return inputs
    def get_raw_train_dataset(self):
        self.train_examples = self.train_examples.shuffle(self.parameters["shuffle_dataset"])
        return self.train_examples.map(
            self.template, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    def get_raw_dev_dataset(self):
        return self.dev_examples.map(
            self.template, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    def on_the_fly_dev_dataset(self, dev_examples):
        dev_examples =  dev_examples.map(
            self.template, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dev_examples = dev_examples.map(
                    self.preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
        return dev_examples.padded_batch(self.parameters["batch_size"], padding_values=0)

    def preprocess_dataset(self,dataset):
        if self.parameters["greedy_padding"]:
            preprocessed_dataset = dataset_preprocessing.greedyBatch_training_input(
                dataset,
                self.parameters,
                min_boundary=8,
                filter_min=1,
                filter_max=self.parameters["max_sequence_length"],
                preprocess_fn=self.preprocess_fn,
                postprocess_fn=self.postprocess_fn,
            )
        else:
                training_dataset = dataset.map(
                    self.preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
                training_dataset = training_dataset.shuffle(self.parameters["shuffle_dataset"])
                preprocessed_dataset = training_dataset.padded_batch(self.parameters["batch_size"], padding_values=0)
        return preprocessed_dataset
    def preprocessed_training_dataset(self):
        preprocessed_dataset = self.preprocess_dataset(self.get_raw_train_dataset())
        return preprocessed_dataset.with_options(self.data_opt)
    def preprocessed_dev_dataset(self):
        if self.dev_examples is not None:
            preprocessed_dataset = self.preprocess_dataset(self.get_raw_dev_dataset())
            return preprocessed_dataset.with_options(self.data_opt)
        else:
            return None


class DatasetManager_monolingual_LangIDatatSent(DatasetManager):
    def template(self, inputs):
        inputs = tf.strings.split(inputs, "@@")
        inputs = tf.cast(tf.strings.to_number(tf.strings.split(inputs)), tf.int32)
        return inputs


class DatasetManager_monolingual_LangIDatatSent_Classification(DatasetManager):
    def template(self, x,y):
        return (x,y)
