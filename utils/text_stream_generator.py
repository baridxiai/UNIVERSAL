# -*- coding: utf-8 -*-
# code warrior: Barid

from mosestokenizer import MosesTokenizer

import tokenizers

LANG = "en"
STREAM_SIZE = 256
en_tokenize = MosesTokenizer(LANG)
de_tokenize = MosesTokenizer("de")
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
    "/home/vivalavida/massive_data/data/en_wiki.TOK.BPE-STREAM",
    # "/home/vivalavida/massive_data/data/de_wiki.TOK.BPE-STREAM",
]
output_globle = []
for k, v in enumerate(data):
    with open(v, "r") as input:
        with open(v + "-256", "w") as output:
            str = []
            for index, line in enumerate(input.readlines()):
                # t = " ".join(en_tokenize.tokenize(line)) + " " + "[EOS]" + " "
                # str += t
                # num_token += 1
                # num_token += len(en_tokenize.tokenize(line))
                # output.write(" ".join(str[:STREAM_SIZE]))
                # output.write("\n")
                # str = []
                str = str + line.strip().split(" ")
                if len(str) >= STREAM_SIZE:
                    str = str[:STREAM_SIZE]
                    output_globle.append(str)
                    str = []

                if len(output_globle) > 10000:
                    for index, line in enumerate(output_globle):
                        # t = " ".join(en_tokenize.tokenize(line)) + " " + "[EOS]" + " "
                        # re += t
                        # num_token += 1
                        # num_token += len(en_tokenize.tokenize(line))
                        # ids_line = bpe_tok.encode(" ".join(de_tokenize(line))).ids
                        # re = re + [str(ids_line[i]) for i in range(len(ids_line))] + ["2"]
                        # if len(re) >= STREAM_SIZE:
                        output.write(" ".join(line))
                        output.write("\n")
                        output_globle = []