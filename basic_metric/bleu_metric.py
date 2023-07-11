# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import collections
from UNIVERSAL.utils import misc_util
import math
import numpy as np
import re
import six
import sys
import unicodedata
from sacrebleu import BLEU

bleu = BLEU()
# bleu.effective_order = True
# tf.print(bleu.get_signature())


def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


# def compute_bleu(raw_reference_corpus, raw_translation_corpus, eos_id=1, max_order=4, use_bp=True):
#     """Computes BLEU score of translated segments against one or more references.
#   Args:
#     reference_corpus: list of references for each translation. Each
#         reference should be tokenized into a list of tokens.
#     translation_corpus: list of translations to score. Each translation
#         should be tokenized into a list of tokens.
#     max_order: Maximum n-gram order to use when computing BLEU score.
#     use_bp: boolean, whether to apply brevity penalty.
#   Returns:
#     BLEU score.
#   """
#     try:
#         reference_corpus = raw_reference_corpus.numpy().tolist()
#         translation_corpus = raw_translation_corpus.numpy().tolist()
#     except Exception:
#         # eos_id = eos_id
#         # eos_id = 1
#         reference_corpus = raw_reference_corpus
#         translation_corpus = raw_translation_corpus
#     num = 0
#     bleu = 0
#     for (references, translations) in zip(reference_corpus, translation_corpus):
#         reference_length = 0
#         translation_length = 0
#         bp = 1.0
#         geo_mean = 0

#         matches_by_order = [0] * max_order
#         possible_matches_by_order = [0] * max_order
#         precisions = []
#         # if eos_id > 5000:
#         references = misc_util.token_trim(references, eos_id, remider=1)
#         translations = misc_util.token_trim(translations, eos_id, remider=1)
#         # references = data_manager.decode(references).split(' ')
#         # translations = data_manager.decode(translations).split(' ')
#         reference_length += len(references)
#         translation_length += len(translations)
#         ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
#         translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

#         overlap = dict(
#             (ngram, min(count, translation_ngram_counts[ngram])) for ngram, count in ref_ngram_counts.items()
#         )

#         for ngram in overlap:
#             matches_by_order[len(ngram) - 1] += overlap[ngram]
#         for ngram in translation_ngram_counts:
#             possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]

#         precisions = [0] * max_order
#         smooth = 1.0

#         for i in range(0, max_order):
#             if possible_matches_by_order[i] > 0:
#                 precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
#                 if matches_by_order[i] > 0:
#                     precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
#                 else:
#                     smooth *= 2
#                     precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
#             else:
#                 precisions[i] = 0.0

#         if max(precisions) > 0:
#             p_log_sum = sum(math.log(p) for p in precisions if p)
#             geo_mean = math.exp(p_log_sum / max_order)

#         if use_bp:
#             ratio = translation_length / reference_length
#             bp = math.exp(1 - 1.0 / ratio) if ratio < 1.0 else 1.0
#         bleu += geo_mean * bp
#         num += 1
#     if num == 0:
#         return 0, tf.constant(1.0)
#     return np.float32(bleu * 100 / num), tf.constant(1.0)


def compute_bleu(raw_reference_corpus, raw_translation_corpus, eos_id=2, max_order=4, use_bp=True):
    try:
        reference_corpus = raw_reference_corpus.numpy().tolist()
        translation_corpus = raw_translation_corpus.numpy().tolist()
    except Exception:
        #     # eos_id = eos_id
        #     # eos_id = 1
        reference_corpus = raw_reference_corpus
        translation_corpus = raw_translation_corpus
    reference_corpus = [
        " ".join([str(r) for r in misc_util.token_trim(reference, eos_id)]) for reference in reference_corpus
    ]
    translation_corpus = [
        " ".join([str(r) for r in misc_util.token_trim(translation, eos_id)]) for translation in translation_corpus
    ]
    try:
        return round(bleu.corpus_score(translation_corpus, [reference_corpus]).score, 2)
    except Exception:
        return 0.0


def approx_bleu(labels, logits, trim_id=0):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.argmax(logits, axis=-1)
    else:
        logits = tf.cast(logits, tf.int64)
    labels = tf.cast(labels, tf.int64)
    logits = tf.cast(logits, tf.int64)
    score = tf.py_function(compute_bleu, [labels, logits, trim_id], tf.float32)
    return score, tf.constant(1.0)


def approx_unigram_bleu(labels, logits):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.argmax(logits, axis=-1)
    else:
        logits = tf.cast(logits, tf.int64)
    labels = tf.cast(labels, tf.int64)
    score = tf.py_function(compute_unigram_bleu, [labels, logits], tf.float32)
    return score, tf.constant(1.0)


def compute_unigram_bleu(labels, logits):
    return compute_bleu(labels, logits, max_order=1)


def unigram_bleu_fn(labels, logits):
    return approx_unigram_bleu(labels, logits)[0]


def bleu_fn(labels, logits):
    return compute_bleu(labels, logits)[0]


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols."""

    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(
            six.unichr(x) for x in range(sys.maxunicode) if unicodedata.category(six.unichr(x)).startswith(prefix)
        )


class Unigram_BLEU_Metric(tf.keras.layers.Layer):
    def __init__(self, name="Unigram_BLEU"):
        self.mean = tf.keras.metrics.Mean(name)
        super(Unigram_BLEU_Metric, self).__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = approx_unigram_bleu(y_true, y_pred)
        value = self.mean(value)
        self.add_metric(value)
        return y_pred


class Quadrugram_BLEU_Metric(tf.keras.layers.Layer):
    def __init__(self, name):
        self.mean = tf.keras.metrics.Mean(name)
        super(Quadrugram_BLEU_Metric, self).__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = approx_bleu(y_true, y_pred)
        value = self.mean(value)
        self.add_metric(value)
        return y_pred


class FILE_BLEU(object):
    def __init__(self):
        self.uregex = UnicodeRegex()

    def bleu_tokenize(self, string):
        r"""Tokenize a string following the official BLEU implementation.
      See https://github.com/moses-smt/mosesdecoder/'
               'blob/master/scripts/generic/mteval-v14.pl#L954-L983
      In our case, the input string is expected to be just one line
      and no HTML entities de-escaping is needed.
      So we just tokenize on punctuation and symbols,
      except when a punctuation is preceded and followed by a digit
      (e.g. a comma/dot as a thousand/decimal separator).
      Note that a numer (e.g. a year) followed by a dot at the end of sentence
      is NOT tokenized,
      i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
      does not match this case (unless we add a space after each sentence).
      However, this error is already in the original mteval-v14.pl
      and we want to be consistent with it.
      Args:
        string: the input string
      Returns:
        a list of tokens
      """
        string = self.uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
        string = self.uregex.punct_nondigit_re.sub(r" \1 \2", string)
        string = self.uregex.symbol_re.sub(r" \1 ", string)
        return string.split()

    def call(self, ref_filename, hyp_filename, case_sensitive=True):
        """Compute BLEU for two files (reference and hypothesis translation)."""
        ref_lines = tf.io.gfile.GFile(ref_filename).read().strip().splitlines()
        hyp_lines = tf.io.gfile.GFile(hyp_filename).read().strip().splitlines()
        # ref_lines = ['I like cat']
        # hyp_lines = ['I like dog']

        if len(ref_lines) != len(hyp_lines):
            raise ValueError("Reference and translation files have different number of " "lines.")
        if not case_sensitive:
            ref_lines = [x.lower() for x in ref_lines]
            hyp_lines = [x.lower() for x in hyp_lines]
        ref_tokens = [self.bleu_tokenize(x) for x in ref_lines]
        hyp_tokens = [self.bleu_tokenize(x) for x in hyp_lines]
        return compute_bleu(ref_tokens, hyp_tokens) * 100
