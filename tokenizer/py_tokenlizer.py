# -*- coding: utf-8 -*-
# code warrior: Barid
from mosestokenizer import MosesTokenizer


def main():
    en_tokenize = MosesTokenizer("de")
    en_copurs = [
        "/home/vivalavida/massive_data/data/fair/wmt14_en_de/train.tags.en-de.tok.de",
    ]
    for k, v in enumerate(en_copurs):
        with open(v, 'r') as input:
            with open(v + ".TOK", 'w') as output:
                for index, line in enumerate(input.readlines()):
                    output.write(" ".join(en_tokenize.tokenize(line)))
                    output.write("\n")


if __name__ == "__main__":
    main()
