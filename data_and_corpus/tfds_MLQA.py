# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
from mosestokenizer import MosesTokenizer
import os.path
gcs_utils._is_gcs_disabled = True

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results
def MLQA_tfds(language, tokenizer,lang_dict, BOS,EOS):
    mlqa_file = "./mlqa_"+ "_".join(language) +".corpurs"
    if os.path.isfile(mlqa_file):
        pass
    else:
        with open(mlqa_file,"w") as file:
            for lang in language:
                mose = MosesTokenizer(lang.lower())
                mlqa = "mlqa/" + lang.lower()
                mlqa = tfds.load(mlqa,try_gcs=True)
                mlqa_test = mlqa["test"]
                for d in tfds.as_numpy(mlqa_test):
                    # answer_start = d["answer"]["answer_start"][0]
                    answer = d["answer"]["text"].decoder()[0]
                    context = d["context"].decode()
                    question = d["question"].decode()
                    answer = tokenizer.encode(mose(answer), is_pretokenized=True)
                    context = [str(BOS)] + tokenizer.encode(mose(context), is_pretokenized=True) + [str(EOS)]
                    question = tokenizer.encode(mose(question), is_pretokenized=True) + [str(EOS)]
                    answer_start,answer_end = find_sub_list(answer,context)
                    data = lang + "@@" + lang_dict[lang] + "@@" + str(answer_start) + "@@" + str(answer_end) + "@@" + " ".join(context+question)
                file.write(data)
                file.write("/n")
