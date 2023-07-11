# -*- coding: utf-8 -*-
# code warrior: Barid
from mosestokenizer import MosesTokenizer


def main():
    en_tokenize = MosesTokenizer("es")
    en_copurs = [
        "/home/vivalavida/massive_data/data/tr_wiki",
    ]
    for k, v in enumerate(en_copurs):
        with open(v, "r") as input:
            with open(v + ".TOK", "w") as output:
                for index, line in enumerate(input.readlines()):
                    output.write(" ".join(en_tokenize(line)))
                    output.write("\n")


# def main():
#     from pythainlp import word_tokenize

#     copurs = [
#         "/home/vivalavida/massive_data/data/th_wiki",
#     ]
#     for k, v in enumerate(copurs):
#         with open(v, "r") as input:
#             with open(v + ".TOK", "w") as output:
#                 for index, line in enumerate(input.readlines()):
#                     output.write(" ".join(word_tokenize(line, keep_whitespace=False)))
#                     output.write("\n")


if __name__ == "__main__":
    main()
