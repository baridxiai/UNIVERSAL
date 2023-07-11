#!/bin/bash

CORPORA="en fr es de el bg ru tr ar vi th zh hi sw ur"


bpe=xlm15_codes_320k
echo "pre-processing train data..."
./fast learnbpe 320000 ~/massive_data/data/wiki/bpe_learning > ${bpe}
for l in ar bg de el en es fr hi ru sw th tr ur vi zh; do
    mkdir -p ./wiki_tmp/${bpe}/
    echo ~/massive_data/data/wiki/${l}_wiki.TOK
    echo ./wiki_tmp/${bpe}_${l}_bpe
    ./fast applybpe ./wiki_tmp/${bpe}/${bpe}_${l}_bpe ~/massive_data/data/wiki/${l}_wiki.TOK ${bpe}
    ./fast getvocab ./wiki_tmp/${bpe}/${bpe}_${l}_bpe > ./wiki_tmp/${bpe}/${bpe}_${l}_vocab
done
bpe=xlm15_codes_20k
echo "pre-processing train data..."
./fast learnbpe 20000 ~/massive_data/data/wiki/bpe_learning > ${bpe}
for l in ar bg de el en es fr hi ru sw th tr ur vi zh; do
    mkdir -p ./wiki_tmp/${bpe}/
    echo ~/massive_data/data/wiki/${l}_wiki.TOK
    echo ./wiki_tmp/${bpe}_${l}_bpe
    ./fast applybpe ./wiki_tmp/${bpe}/${bpe}_${l}_bpe ~/massive_data/data/wiki/${l}_wiki.TOK ${bpe}
    ./fast getvocab ./wiki_tmp/${bpe}/${bpe}_${l}_bpe > ./wiki_tmp/${bpe}/${bpe}_${l}_vocab
done

# bpe=xlm15_codes_160k
# echo "pre-processing train data..."
# for l in ar bg de el en es fr hi ru sw th tr ur vi zh; do
#     mkdir -p ./wiki_tmp/${bpe}/
#     echo ~/massive_data/data/wiki/${l}_wiki.TOK
#     echo ./wiki_tmp/${bpe}_${l}_bpe
#     ./fast applybpe ./wiki_tmp/${bpe}/${bpe}_${l}_bpe ~/massive_data/data/wiki/${l}_wiki.TOK ${bpe}
#     ./fast getvocab ./wiki_tmp/${bpe}/${bpe}_${l}_bpe > ./wiki_tmp/${bpe}/${bpe}_${l}_vocab
# done

# bpe=xlm15_codes_40k
# echo "pre-processing train data..."
# for l in ar bg de el en es fr hi ru sw th tr ur vi zh; do
#     mkdir -p ./wiki_tmp/${bpe}/
#     echo ~/massive_data/data/wiki/${l}_wiki.TOK
#     echo ./wiki_tmp/${bpe}_${l}_bpe
#     ./fast applybpe ./wiki_tmp/${bpe}/${bpe}_${l}_bpe ~/massive_data/data/wiki/${l}_wiki.TOK ${bpe}
#     ./fast getvocab ./wiki_tmp/${bpe}/${bpe}_${l}_bpe > ./wiki_tmp/${bpe}/${bpe}_${l}_vocab
# done