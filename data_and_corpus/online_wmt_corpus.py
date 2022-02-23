# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import tensorflow_datasets as tfds
import fasttext
lang_checker = fasttext.load_model('../lid.176.ftz')

def online_dataset(src,tgt, wmt_year=2016, parallel=False, path="~/"):
    if src == 'fr':
        testset = "newstest2014"
    else:
        if src == 'fr':
            testset = "newstest2017"
        else:
            testset = "newstest2016"
    trainset = [
        # "casia2015", "casict2011", "casict2015", "datum2015", "datum2017", "neu2017",
        "europarl_v7",
        "europarl_v9",
        "europarl_v8_16",
        "commoncrawl",
        "newscommentary_v11",
        "newscommentary_v10",
        "newscommentary_v12",
        "newscommentary_v13",
        "newscommentary_v14",
        "paracrawl_v1",
        "czeng_16pre",
        "wikiheadlines_fi",
        "wikiheadlines_ru",
        "setimes_2",
        "paracrawl_v1_ru",
        "uncorpus_v1"
    ]
    config = tfds.translate.wmt.WmtConfig(
        version=tfds.core.Version(
            '0.0.1', experiments={tfds.core.Experiment.S3: False}),
        language_pair=(src, tgt),
        subsets={
            tfds.Split.TRAIN: trainset,
            tfds.Split.TEST: [testset],
        },
    )
    builder = tfds.builder("wmt_translate",
                           config=config,
                           data_dir=path + '/corpus/'+ wmt_year)
    builder.download_and_prepare(download_dir=path + '/corpus/'+ wmt_year)
    examples = builder.as_dataset(as_supervised=True)
    train_examples = examples['train']
    test_examples = examples['test']
    print('downloaded')

    if tf.io.gfile.exists(path + '/'+ wmt_year+'_TRAINING_monolingual_' + src
                          + '.corpus') is not True:

        with tf.io.gfile.GFile(path + '/'+ wmt_year+'_TRAINING_monolingual_' + src
                               + '.corpus',
                               mode="w") as f_src:
            with tf.io.gfile.GFile(path + '/'+ wmt_year+'_TRAINING_monolingual_' + tgt
                                   + '.corpus',
                                   mode="w") as f_tgt:
                for src, tgt in train_examples:
                    s_0 = src.numpy().decode()
                    s_1 = tgt.numpy().decode()
                    s_0_lang, s_0_p = lang_checker.predict(s_0, k=2)
                    s_1_lang, s_1_p = lang_checker.predict(s_1, k=2)
                    if s_0[0] != '{' and s_0_lang[
                            0] == '__label__'+src and s_0_p[0] > 0.9 and s_0[
                                -1] != '}' :
                        f_src.write("%s\n" % s_0)
                    if s_1[0] != '{' and s_1_lang[
                            0] == '__label__'+tgt and s_1_p[0] > 0.9 and s_1[
                                -1] != '}' :
                        f_tgt.write("%s\n" % s_1)
    if  parallel:
        if tf.io.gfile.exists(path + '/'+ wmt_year+'_TESTING_' + src
                              + '.corpus') is not True:

            with tf.io.gfile.GFile(path + '/'+ wmt_year+'_TESTING_' + src
                                   + '.corpus',
                                   mode="w") as f_src:
                with tf.io.gfile.GFile(path + '/'+ wmt_year+'_TESTING_' + tgt
                                       + '.corpus',
                                       mode="w") as f_tgt:
                    for src, tgt in test_examples:
                        s_0 = src.numpy().decode()
                        s_1 = tgt.numpy().decode()
                        s_0_lang, s_0_p = lang_checker.predict(s_0, k=2)
                        s_1_lang, s_1_p = lang_checker.predict(s_1, k=2)
                        if s_0[0] != '{' and s_0_lang[
                                0] == '__label__'+src and s_0_p[0] > 0.9 and s_0[
                                    -1] != '}' :
                            f_src.write("%s\n" % s_0)
                        if s_1[0] != '{' and s_1_lang[
                                0] == '__label__'+tgt and s_1_p[0] > 0.9 and s_1[
                                    -1] != '}' :
                            f_tgt.write("%s\n" % s_1)
