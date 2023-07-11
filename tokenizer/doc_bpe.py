# -*- coding: utf-8 -*-
# code warrior: Barid
# import sentencepiece as sp
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split
from tokenizers import pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFD, StripAccents, Nmt
import numpy as np
import six
import sys
import unicodedata
import os

c_path = os.getcwd()  # "no last slash"
data = [
    # "/home/vivalavida/massive_data/data/Tokenized.news.2015.de.shuffled_filter.TEXT-STREAM-200",
    "/home/vivalavida/massive_data/data/Tokenized.news.2008.de.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2017.de.shuffled_filter.TEXT-STREAM-200",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2015.en.shuffled_filter.TEXT-STREAM-200",
    "/home/vivalavida/massive_data/data/Tokenized.news.2007.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2017.en.shuffled_filter.TEXT-STREAM-200",
]
# sp.SentencePieceTrainer.train(input=data,
#                               model_prefix='bilingual_deen60000',
#                               vocab_size=60000,
#                               model_type='bpe',
#                               bos_id=1,
#                               eos_id=2,
#                               unk_id=3,
#                               pad_id=0)
tokenizer = Tokenizer(BPE(continuing_subword_prefix="@@"))
# normalizer = normalizers.Sequence([NFD(), StripAccents(),Nmt()])tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.pre_tokenizer = Split(".", "removed")
trainer = BpeTrainer(
    special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]"],
    vocab_size=60000,
    min_frequency=10000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    continuing_subword_prefix="@@",
)
tokenizer.train(files=data, trainer=trainer)
tokenizer.save(c_path + "/vocabulary/DeEn_60000_mono/tokenizer.json")
files = tokenizer.model.save(c_path + "/vocabulary/DeEn_100000_doc/", c_path + "/vocabulary/DeEn_100000_doc/",)
