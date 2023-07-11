#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:36:55 2016

@author: Barid
"""

import os
import argparse
import fileinput

parser = argparse.ArgumentParser(description="Merge multiple files to one")
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()
input_dir = args.input_dir
output_file = args.output_file
# Open a file
# path = "/Users/xiai/Documents/workspace/wifianalysis/csv_raw"
dirs = os.listdir(input_dir)
# This would print all the files and directories

with open(output_file, "w") as output_f:
    for sub_dir in dirs:
        sub_sub_dir = os.listdir(input_dir + "/" + sub_dir)
        sub_sub_dir = [input_dir + "/" + sub_dir + "/" + file for file in sub_sub_dir]
        for line in fileinput.input(sub_sub_dir):
            line = line.strip()
            if len(line) > 1:
                if line[0] != "<":
                    output_f.write(line)
                    output_f.write("\n")
