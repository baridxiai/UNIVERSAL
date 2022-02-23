# -*- coding: utf-8 -*-
# code warrior: Barid
#
# import sentencepiece as sp
# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
from tokenizers import Tokenizer
from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
# data = [
#     ]
data = [
    "/home/vivalavida/massive_data/data/news.2007.fr.shuffled.TOKENIZED",
    "/home/vivalavida/massive_data/data/news.2008.fr.shuffled.TOKENIZED",
    "/home/vivalavida/massive_data/data/news.2009.fr.shuffled.TOKENIZED",
    "/home/vivalavida/massive_data/data/news.2010.fr.shuffled.TOKENIZED",
    "/home/vivalavida/massive_data/data/news.2011.fr.shuffled.TOKENIZED",
    "/home/vivalavida/massive_data/data/news.2012.fr.shuffled.TOKENIZED",
    "/home/vivalavida/massive_data/data/news.2013.fr.shuffled.TOKENIZED",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2007.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2008.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2009.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2010.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2011.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2012.en.shuffled_filter",
    # "/home/vivalavida/massive_data/data/Tokenized.news.2013.en.shuffled_filter",
    ]

# sp.SentencePieceTrainer.train(input=data,
#                               model_prefix='bilingual_deen60000',
#                               vocab_size=60000,
#                               model_type='bpe',
#                               bos_id=1,
#                               eos_id=2,
#                               unk_id=3,
#                               pad_id=0)
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
files = tokenizer.model.save(
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/FrEn_60000",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/FrEn_60000")
tokenizer.model = BPE.from_file(
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/FrEn_60000/vocab.json",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/FrEn_60000/merges.txt",
    unk_token="[UNK]")
total_num = 0
monolingual_vocab = dict()
for d in data:
    with open(d, "r") as f:
        for _, v in enumerate(f.readlines()):
            total_num += 1
            for i in tokenizer.encode(v.strip()).ids:
                if i in monolingual_vocab:
                    monolingual_vocab[i] += 1
                else:
                    monolingual_vocab[i] = 1
        f.close()
        print(total_num)
with open(
        "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/FrEn_60000/statistic_monolingual_Fr60000.statistic",
        "w") as f:
    for k, v in enumerate(
            sorted(monolingual_vocab.items(),
                   key=lambda item: item[1],
                   reverse=True)):
        # f_src.write("%s\n" % k)
        f.write("%d@%d\n" % (v[0], v[1]))
print(total_num)

# tokenizer.save("./vocabulary/DeEn_60000/tokenizer_config_in_case.json")
# tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")
# output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
# print(output.tokens)
# print(output.ids)

data = [
    "/home/vivalavida/massive_data/data/Tokenized.news.2007.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2008.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2009.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2010.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2011.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2012.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2013.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2014.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2015.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2016.en.shuffled_filter",
    "/home/vivalavida/massive_data/data/Tokenized.news.2017.en.shuffled_filter",
]
total_num = 0
monolingual_vocab = dict()
for d in data:
    with open(d, "r") as f:
        for _, v in enumerate(f.readlines()):
            total_num += 1
            for i in tokenizer.encode(v.strip()).ids:
                if i in monolingual_vocab:
                    monolingual_vocab[i] += 1
                else:
                    monolingual_vocab[i] = 1
        f.close()
        print(total_num)
with open(
        "/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/FrEn_60000/statistic_monolingual_En60000.statistic",
        "w") as f:
    for k, v in enumerate(
            sorted(monolingual_vocab.items(),
                   key=lambda item: item[1],
                   reverse=True)):
        # f_src.write("%s\n" % k)
        f.write("%d@%d\n" % (v[0], v[1]))
print(total_num)
