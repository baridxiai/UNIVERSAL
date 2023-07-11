# -*- coding: utf-8 -*-
# code warrior: Barid

# import sentencepiece as sp
from tokenizers import Tokenizer
import tensorflow as tf

from tokenizers.models import WordLevel
from mosestokenizer import MosesTokenizer
import os
from tokenizers.pre_tokenizers import Whitespace

vocab = WordLevel.read_file(
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/bi_vocab.json"
)
bpe_tok = WordLevel(vocab, unk_token="[UNK]")
bpe_tok = Tokenizer(bpe_tok)
# bpe_tok = Tokenizer.from_file("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/tokenizer.json")
data = [
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/mono_vocab_de",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/mono_vocab_en",
]
for k, v in enumerate(data):
    total = 0
    with open(v, "r") as input:
        with open(v + ".ids", "w") as output:
            for index, line in enumerate(input.readlines()):
                word, count = line.strip().split(" ")
                total += int(count)
                ids = bpe_tok.encode(word.strip()).ids[0]
                # try:
                #     ids = bpe_tok.encode(word).ids[0]
                # except Exception:
                #     import pdb;pdb.set_trace()
                #     print(word)
                output.write(str(ids) + " " + str(count))
                output.write("\n")
    print(total)
