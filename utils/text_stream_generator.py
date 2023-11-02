# -*- coding: utf-8 -*-
# code warrior: Barid

from mosestokenizer import MosesTokenizer

import tokenizers

LANG = "en"
STREAM_SIZE = 256
data = [
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/en_wiki_1G",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/fr_wiki_1G",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/de_wiki_1G",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/ru_wiki_1G",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/zh_wiki_1G",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/sw_wiki_1G",
    "/home/vivalavida/workspace/alpha/UNIVERSAL/tokenizer/fastBPE/xlm7-150k/ur_wiki_1G",
]
output_globle = []
for k, v in enumerate(data):
    with open(v, "r") as input:
        with open(v + "-256", "w") as output:
            str = ["[EOS]"]
            for index, line in enumerate(input.readlines()):
                str = str + line.strip().split(" ") + ["[EOS]"]
                if len(str) >= STREAM_SIZE:
                    str = str[:STREAM_SIZE-1] + ["[EOS]"]
                    output_globle.append(str)
                    str = ["[EOS]"]

                if len(output_globle) > 10000:
                    for index, line in enumerate(output_globle):
                        # if len(re) >= STREAM_SIZE:
                        output.write(" ".join(line))
                        output.write("\n")
                        output_globle = []