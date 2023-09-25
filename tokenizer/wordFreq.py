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
    "/Users/barid/Documents/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_6K_vocab.json"
)
bpe_tok = WordLevel(vocab, unk_token="[UNK]")
bpe_tok = Tokenizer(bpe_tok)
# bpe_tok = Tokenizer.from_file("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/tokenizer.json")
data = [
    ("/Users/barid/Documents/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K.en.vocab",201385103),
    ("/Users/barid/Documents/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K.de.vocab",216757090),
    ("/Users/barid/Documents/workspace/alpha/UNIVERSAL/vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K.hi.vocab",66931677),
]
for k, v in enumerate(data):
    total = 0
    with open(v[0], "r") as input:
        with open(v[0] + ".freq", "w") as output:
            for index, line in enumerate(input.readlines()):
                word, count = line.strip().split(" ")
                total += int(count)
                ids = bpe_tok.encode(word.strip()).ids[0]
                # try:
                #     ids = bpe_tok.encode(word).ids[0]
                # except Exception:
                #     import pdb;pdb.set_trace()
                #     print(word)
                output.write(str(word)+ " " + str(ids) + " " + str(int(count)/v[1]))
                output.write("\n")
    print(total)
