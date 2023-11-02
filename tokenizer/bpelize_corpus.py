# -*- coding: utf-8 -*-
# code warrior: Barid

# import sentencepiece as sp
from tokenizers import Tokenizer
import tensorflow as tf

from tokenizers.models import WordLevel
from mosestokenizer import MosesTokenizer
import os
from tokenizers.pre_tokenizers import Whitespace

cwd = os.getcwd()


def percent(current, total):
    import sys

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = "#" * int(current / total * bar_length)
    spaces = " " * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total)))


LANG = "de"
STREAM_SIZE = 256
# bpe_tok = Tokenizer.from_file(cwd + "/../UNIVERSAL/vocabulary/DeEn_6k_wiki_balanced/tokenizer.json")
# bpe_tok = Tokenizer.from_file(cwd + "/../UNIVERSAL/vocabulary/DeEn_6k_wiki_balanced/tokenizer.json")
vocab = WordLevel.read_file(
    "./xlm7-150k/vocab.json"
)
bpe_tok = WordLevel(vocab, unk_token="[UNK]")
bpe_tok = Tokenizer(bpe_tok)
# bpe_tok.pre_tokenizer = Whitespace()
en_tokenize = MosesTokenizer(LANG)
de_tokenize = MosesTokenizer("en")
data = [
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/en_wiki_1G-256",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/fr_wiki_1G-256",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/de_wiki_1G-256",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/ru_wiki_1G-256",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/zh_wiki_1G-256",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/sw_wiki_1G-256",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/ur_wiki_1G-256",
]
output_globle = []
suffix = "._vocab"
truncate = True
for k, v in enumerate(data):
    with open(v, "r") as input:
        with open(v + suffix, "w") as output:
            re = []
            num_token = 0
            for index, line in enumerate(input.readlines()):
                # t = " ".join(en_tokenize.tokenize(line)) + " " + "[EOS]" + " "
                # re += t
                # num_token += 1
                # num_token += len(en_tokenize.tokenize(line))
                ids_line = bpe_tok.encode(line.strip().split(" "), is_pretokenized=True).ids
                re = [str(ids_line[i]) for i in range(len(ids_line))]
                    # output_globle.append(re)
                re = " ".join(re)
                # re = re.replace("3 41535","2")
                if truncate:
                    re = re[:STREAM_SIZE]
                output.write(str(k+1) + "@@" + re)
                output.write("\n")
