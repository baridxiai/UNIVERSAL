# -*- coding: utf-8 -*-
# code warrior: Barid
# import sentencepiece as sp
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
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

]
tokenizer = Tokenizer(WordPiece(continuing_subword_prefix="@@"))
# normalizer = normalizers.Sequence([NFD(), StripAccents(),Nmt()])tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(
    special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]"],
    vocab_size=60000,
    # min_frequency=1786,
    # initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    continuing_subword_prefix="@@",
)
tokenizer.encode()
tokenizer.train(files=data, trainer=trainer)
tokenizer.save(c_path + "/vocabulary/DeEn_6k_wiki_balanced/tokenizer.json")
files = tokenizer.model.save(
    c_path + "/vocabulary/DeEn_6k_wiki_balanced/", c_path + "/vocabulary/DeEn_6k_wiki_balanced/",
)
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
