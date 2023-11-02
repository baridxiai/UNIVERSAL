

BEST PRACTICE ADVICE FOR BYTE PAIR ENCODING IN NMT

We found that for languages that share an alphabet, learning BPE on the concatenation of the (two or more) involved languages increases the consistency of segmentation, and reduces the problem of inserting/deleting characters when copying/transliterating names.

However, this introduces undesirable edge cases in that a word may be segmented in a way that has only been observed in the other language, and is thus unknown at test time. To prevent this, apply_bpe.py accepts a --vocabulary and a --vocabulary-threshold option so that the script will only produce symbols which also appear in the vocabulary (with at least some frequency).

To use this functionality, we recommend the following recipe (assuming L1 and L2 are the two languages):

Learn byte pair encoding on the concatenation of the training text, and get resulting vocabulary for each:

                                                     ---- from Rico Sennrich
**Learn codes**

./fast learnbpe 40000 train.de train.en > codes


**Apply codes to train**


./fast applybpe train.de.40000 train.de codes
./fast applybpe train.en.40000 train.en codes

**Get train vocabulary**

./fast getvocab train.de.40000 > vocab.de.40000
./fast getvocab train.en.40000 > vocab.en.40000

**Apply codes to train again wiht vocabularies!!!!!!!!**

./fast applybpe train.de.40000 train.de codes vocab.de.40000
./fast applybpe train.en.40000 train.en codes vocab.en.40000

**Apply codes to valid and test**

./fast applybpe valid.de.40000 valid.de codes vocab.de.40000
./fast applybpe valid.en.40000 valid.en codes vocab.en.40000
./fast applybpe test.de.40000  test.de  codes vocab.de.40000
./fast applybpe test.en.40000  test.en  codes vocab.en.40000



################XLM7-150k###############################################################################
###########BPE
Loading codes from xlm7-150k/bpe_code ...
Read 150000 codes from the codes file.
Loading vocabulary from learning_bpe_temp ...
Read 2601927224 words (15777530 unique) from text file.
Applying BPE to learning_bpe_temp ...
Modified 2601927224 words from text file.
###########VOCAB
Loading vocabulary from bpe_temp ...
Read 3056890369 words (189491 unique) from text file.