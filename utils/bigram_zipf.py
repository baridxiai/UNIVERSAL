# -*- coding: utf-8 -*-
# code warrior: Barid

from tokenizers.models import WordLevel
from tokenizers import Tokenizer
import fastBPE
from mosestokenizer import MosesTokenizer
en_token = MosesTokenizer('en')
de_token = MosesTokenizer('de')

from numpy import load
#co_mat = load("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/globle_cooccurrence.npy")
co_mat = load("/home/vivalavida/workspace/alpha/UNIVERSAL/data_n1.npy")
vocab = WordLevel.read_file("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/bi_vocab.json")
vocab = WordLevel(vocab, unk_token="[UNK]")
vocab = Tokenizer(vocab)
bpe_tok = fastBPE.fastBPE("/home/vivalavida/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/bpe_code")
with open("/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/phrase-table-wiki_noNUM_noPUN", 'r') as f:
    with open("/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/bigram-phrase-table-wiki_noNUM_noPUN",'w') as output:
        for _,line in enumerate(f.readlines()):
            de_str,en_str = line.strip().split("99")
            de = de_token(de_str.strip())
            en = en_token(en_str.strip())
            de_bpe = bpe_tok.apply([" ".join(de)])[0].strip()
            en_bpe = bpe_tok.apply([" ".join(en)])[0].strip()
            de = de_bpe.split(" ")
            en = en_bpe.split(" ")
            if len(de) == 2 and len(en) == 2:
                de = vocab.encode(de,is_pretokenized=True).ids
                en = vocab.encode(en,is_pretokenized=True).ids
                if len(de) == 2 and len(en) == 2:
                    de = co_mat[de[0],de[1]]
                    en = co_mat[en[0],en[1]]
                    output.write(str(de_bpe) + "&&" +str(en_bpe) + '##' +str(de)+"&&"+str(en))
                    output.write("\n")
