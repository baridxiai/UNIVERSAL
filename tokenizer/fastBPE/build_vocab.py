#!/usr/bin/env python3

from collections import OrderedDict
import fileinput
import sys

import numpy
import json


def read_vocab(path):
    ret = []
    with open(path, "r") as f:
        for line in f:
            x, y = line.strip().split()[:2]
            x, y = x.replace("</w>", ""), y.replace("</w>", "")
            ret.append((x, y))
    return ret


def check_letters(string, checker):
    for c in string:
        if c in checker:
            pass
        else:
            return False
    return True


def main():
    sorted_words = set()
    merges = read_vocab(sys.argv[1])
    merges = sorted(list(set("".join(x + y for x, y in merges))))
    for filename in sys.argv[2:]:
        print("Processing", filename)
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                words_in, counts = line.strip().split(" ")

                #         for w in words_in:
                #             if w not in word_freqs:
                #                 word_freqs[w] = 0
                #             word_freqs[w] += 1
                # words = list(word_freqs.keys())
                # freqs = list(word_freqs.values())

                # sorted_idx = numpy.argsort(freqs)

                if check_letters(words_in, merges) or int(counts) > 10:
                    sorted_words.add(words_in)

    worddict = OrderedDict()
    worddict["[Pad]"] = 0
    worddict["[SOS]"] = 1
    worddict["[EOS]"] = 2
    worddict["[UNK]"] = 3
    worddict["[MASK]"] = 4
    # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 5

    with open("%s.json" % "final_vocab", "w", encoding="utf-8") as f:
        json.dump(worddict, f, indent=2, ensure_ascii=False)

    print("Done")

with open("vocabulary/EnDeHi_6k_wiki/EnDeHi_codes_6K_all_vocab", "r", encoding="utf-8") as f:
    worddict = OrderedDict()
    worddict["[Pad]"] = 0
    worddict["[SOS]"] = 1
    worddict["[EOS]"] = 2
    worddict["[UNK]"] = 3
    worddict["[MASK]"] = 4
    for k,v in enumerate(f.readlines()):
        words_in, counts = v.strip().split(" ")

        #         for w in words_in:
        #             if w not in word_freqs:
        #                 word_freqs[w] = 0
        #             word_freqs[w] += 1
        # words = list(word_freqs.keys())
        # freqs = list(word_freqs.values())

        # sorted_idx = numpy.argsort(freqs)

        worddict[words_in] = k+5

    # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.

    with open("vocabulary/EnDeHi_6k_wiki/EnDeHi_6K_vocab.json", "w", encoding="utf-8") as f:
        json.dump(worddict, f, indent=2, ensure_ascii=False)

    print("Done")

if __name__ == "__main__":
    main()
