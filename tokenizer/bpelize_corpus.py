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
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/bi_vocab.json"
)
bpe_tok = WordLevel(vocab, unk_token="[UNK]")
bpe_tok = Tokenizer(bpe_tok)
# bpe_tok.pre_tokenizer = Whitespace()
en_tokenize = MosesTokenizer(LANG)
de_tokenize = MosesTokenizer("en")
data = [
    # "/home/vivalavida/massive_data/data/Tokenized.news.2007.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2008.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2009.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2010.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2011.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2012.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2013.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2014.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2015.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2016.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2017.en.shuffled_filter",

    # "/home/vivalavida/massive_data/data/wiki/en_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/de_wiki.TOK",

    # "/home/vivalavida/massive_data/data/wiki/ar_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/bg_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/de_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/el_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/en_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/es_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/fr_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/hi_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/ru_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/sw_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/ur_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/vi_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/zh_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/tr_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/th_wiki.TOK",
    # "/home/vivalavida/massive_data/data/wiki/dev_en",
    # "/home/vivalavida/massive_data/data/wiki/dev_de",

    # "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/EnDeHi.en.12k",
    # "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/EnDeHi.de.12k",
    # "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/EnDeHi.hi.12k",
    "/home/vivalavida/massive_data/data/XNLI/nli_data_de_test_60kDEEN",
    "/home/vivalavida/massive_data/data/XNLI/nli_train_en_60kDEEN",

    # "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/train.de.60000.withVoc"
    # "/home/vivalavida/massive_data/data/XNLI/nli_data_de_bpe",
    # "/home/vivalavida/massive_data/data/XNLI/nli_data_de_test_EnDeHi_bpe",
    # "/home/vivalavida/massive_data/data/XNLI/nli_data_en_test_EnDeHi_bpe",
    # "/home/vivalavida/massive_data/data/XNLI/nli_train_en_30k",
    # "/home/vivalavida/massive_data/data/XNLI/nli_dat_de_test_30k",
]
output_globle = []
suffix = "._vocab"
truncate = True
# for k, v in enumerate(data):
#     n = 0
#     with open(v, "r") as input:
#         with open(v + suffix, "w") as output:
#             num_token = 0
#             re = []
#             for index, line in enumerate(input.readlines()):
#                 ids_line = bpe_tok.encode(line.strip().split(" "), is_pretokenized=True).ids +[2]
#                 r = [str(ids_line[i]) for i in range(len(ids_line))]
#                 re += r
#                     # output_globle.append(re)
#                 if len(re) >= STREAM_SIZE:
#                     if truncate:
#                         re = re[:STREAM_SIZE]
#                     re = str(k+1)+"@@"+" ".join(re)
#                     output.write(re)
#                     output.write("\n")
#                     # num_token = 0
#                     n +=1
#                     re = []
#     print(n)
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
                re = re.replace("3 41535","2")
                if truncate:
                    re = re[:STREAM_SIZE]
                output.write(str(k+1) + "@@" + re)
                output.write("\n")
