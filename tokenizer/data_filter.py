# -*- coding: utf-8 -*-
# code warrior: Barid

import re
import argparse

from langdetect import detect
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
polyglot_logger.setLevel("ERROR")


def get_parser():
    parser = argparse.ArgumentParser(description="Remove noisy data")

    # parser.add_argument("--input", type=str,
    #                     help="The path of input file")
    parser.add_argument("--lang", type=str,
                        help="The language of input file")
    # parser.add_argument("--output", type=str, default=None,
    #                     help="The path of output file")

    return parser


def detect_exist_url(text):
    urls = re.findall(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    url1 = re.findall(
        'http[s]?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls) > 0 or len(url1) > 0


def detect_lang(text, lang):
    try:
        for i, l in enumerate(Detector(text, quiet=True).languages):
            if l.code == lang and i == 0:
                return True
        if detect(text) == lang:
            return True
        return False
    except:
        return False


def main():
    parser = get_parser()
    args = parser.parse_args()

    count = 0
    allcount = 0
    file_list = [
        "/home/vivalavida/massive_data/data/Tokenized.news.2007.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2008.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2009.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2010.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2011.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2012.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2013.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2014.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2015.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2016.en.shuffled",
        "/home/vivalavida/massive_data/data/Tokenized.news.2017.en.shuffled", ]

    for f in file_list:
        f_w = open(f + '_filter', 'w')
        with open(f, encoding='utf-8') as input_file:
            for line in input_file:
                allcount += 1
                line = line.strip()
                if detect_exist_url(line) is False:
                    if detect_lang(line, args.lang) is True:
                        count += 1
                        # if args.output is not None:
                        f_w.write(line + '\n')
                    # print(line)
                if allcount % 1000000 == 0:
                    print("{} sentences processed".format(allcount), count)
            f_w.close()
    print(count, allcount)


if __name__ == "__main__":
    main()
