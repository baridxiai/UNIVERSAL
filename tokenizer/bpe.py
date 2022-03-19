# -*- coding: utf-8 -*-
# code warrior: Barid
# import sentencepiece as sp
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFD, StripAccents, Nmt
import numpy as np
import six
import sys
import unicodedata
import os

c_path = os.getcwd()  # "no last slash"
data = [
    "/home/vivalavida/massive_data/data/fair/wmt14_en_de_v3/train.de",
    "/home/vivalavida/massive_data/data/fair/wmt14_en_de_v3/train.en",
    # "/home/vivalavida/massive_data/data/europarl-v7/europarl-v7.de-en.en",
    # "/home/vivalavida/massive_data/data/training-parallel-commoncrawl/commoncrawl.de-en.en",
    # "/home/vivalavida/massive_data/data/News_Commentary/news-commentary-v9.de-en.en",
    # "/home/vivalavida/massive_data/data/europarl-v7/europarl-v7.de-en.de",
    # "/home/vivalavida/massive_data/data/training-parallel-commoncrawl/commoncrawl.de-en.de",
    # "/home/vivalavida/massive_data/data/News_Commentary/news-commentary-v9.de-en.de",
    # "/home/vivalavida/massive_data/data/parallel_corpora/europarl-v7+commoncrawl+news.de-en.de",
    # "/home/vivalavida/massive_data/data/parallel_corpora/europarl-v7+commoncrawl+news.de-en.en",
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
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]"],
    vocab_size=32000,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    continuing_subword_prefix="@@",
)
tokenizer.train(files=data, trainer=trainer)
tokenizer.save(c_path + "/vocabulary/DeEn_32000_v4/tokenizer.json")
files = tokenizer.model.save(c_path + "/vocabulary/DeEn_32000_v4/", c_path + "/vocabulary/DeEn_32000_v4/",)
# tokenizer.model = BPE.from_file(*files, unk_token="[UNK]")
# for d in data:
#     with open(d +".BPE",'w') as output:
#         with open(d,'r') as corpus:
#             for k,v in enumerate(corpus):
#                 output.write(" ".join(tokenizer.encode(v).tokens))
#                 output.write("\n")
# tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
# output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
# print(output.tokens)
# print(output.ids)
