# -*- coding: utf-8 -*-
# code warrior: Barid
import csv
import tensorflow as tf
import collections
from mosestokenizer import MosesTokenizer
# import sentencepiece as sp
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import pre_tokenizers, decoders, trainers, processors
import tensorflow as tf
import fastBPE

XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
XNLI_LANGS_partitial = [  'de']
LANG = 'de'
from mosestokenizer import MosesTokenizer
tokenizer = MosesTokenizer(LANG)



# def generate_XNLI_examples_devAndTest(filepath):
#     """This function returns the examples in the raw (text) form."""
#     rows_per_pair_id = collections.defaultdict(list)

#     with tf.io.gfile.GFile(filepath) as f:
#       reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
#       for row in reader:
#         if str(row['language']) in XNLI_LANGS_partitial:
#             rows_per_pair_id[row['pairID']].append(row)
#     for i in XNLI_LANGS_partitial:
#         with open("nli_data" + '_' +str(i), "w") as nli:
#             with open('nli_label' + '_' +str(i), "w") as label:
#                 # tokenizer = MosesTokenizer(i)
#                 for row_data in rows_per_pair_id.items():
#                     row = row_data[1][0]
#                     if str(row['language']) == i:
#                         # premise =  row['sentence1_tokenized']
#                         # hypothesis = row['sentence2_tokenized']
#                         premise =  row['premise']
#                         hypothesis = row['hypothesis']
#                         if row['gold_label'] == "neutral":
#                             gold_label = 0
#                         elif row['gold_label'] == "contradiction":
#                             gold_label = 1
#                         else:
#                             gold_label = 2
#                         nli.write(str(premise) + " ## " + str(hypothesis)  + " ## ")
#                         nli.write("\n")
#                         label.write(str(gold_label))
#                         label.write("\n")
#         nli.close()
#         label.close()
def generate_XNLI_examples_Train(filepath):
    """This function returns the examples in the raw (text) form."""
    vocab = WordLevel.read_file("/Users/barid/Documents/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/bi_vocab.json")
    bpe_tok = WordLevel(vocab, unk_token="[UNK]")
    vocab_tokenizer = Tokenizer(bpe_tok)
    bpe_tok = fastBPE.fastBPE("/Users/barid/Documents/workspace/alpha/UNIVERSAL/vocabulary/DeEn_6k_wiki_fastBPE/bpe_code")
    def _encode(string):
        string = bpe_tok.apply([string])[0]
        return vocab_tokenizer.encode(string.strip().split(" "), is_pretokenized=True).ids
    rows_per_pair_id = []
    with tf.io.gfile.GFile(filepath) as f:
        # reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        # for row in reader:
        #     if str(row['language']) in XNLI_LANGS_partitial:
        #         rows_per_pair_id.append(row)
        rows_per_pair_id = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    with open("nli_train" + '_' + LANG, "w") as nli:
        with open('nli_train_label' + '_' + LANG, "w") as label:
    # with open("nli_test" + '_' + LANG, "w") as nli:
    #     with open('nli_test_label' + '_' + LANG, "w") as label:
            # tokenizer = MosesTokenizer(i)
            for row in rows_per_pair_id:
                # premise =  row['sentence1_tokenized']
                # hypothesis = row['sentence1_tokenized']
                # premise = _encode(" ".join(tokenizer(str(row['premise']))))
                # hypothesis = _encode(" ".join(tokenizer(str(row['hypo']))))
                premise = _encode(row['premise']) + [2]
                hypothesis =_encode(row['hypo']) + [2]
                # premise =  _encode(row['sentence1_tokenized']) + [2]
                # hypothesis =  _encode(row['sentence2_tokenized']) + [2]
                preHypo = premise  + hypothesis
                if len(preHypo) >255:
                    preHypo = preHypo[:255] + [2]
                if row['label'] == "neutral":
                    gold_label = 0
                elif row['label'] == "contradictory":
                    gold_label = 1
                else:
                    gold_label = 2
                # if row['gold_label'] == "neutral":
                #     gold_label = 0
                # elif row['gold_label'] == "contradiction":
                #     gold_label = 1
                # else:
                #     gold_label = 2
                nli.write("2@@2 " + " ".join([str(p) for p in preHypo]))
                nli.write("\n")
                label.write(str(gold_label))
                label.write("\n")
    nli.close()
    label.close()
# generate_XNLI_examples_Train("/Users/barid/Downloads/XNLI-1.0/xnli.test.tsv")
generate_XNLI_examples_Train("/Users/barid/Desktop/XNLI-MT-1.0/multinli/multinli.train.en.tsv")
                    # re_list.append((row['language'],str(premise) + " " + str(hypothesis), gold_label))
      # rows[0]['pairID'], {
      #     'premise': premise,
      #     'hypothesis': hypothesis,
      #     'label': gold_label
      # }
