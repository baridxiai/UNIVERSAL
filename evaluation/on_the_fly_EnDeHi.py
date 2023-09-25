from UNIVERSAL.evaluation.get_test_data import get_SemEval
from UNIVERSAL.basic_metric import embedding_space
import tensorflow as tf
def on_the_fly_EnDeHi(data_manager,model):
    def _token2id(tokens):
            ids = [ t.ids for t in data_manager.tokenizer.encode_batch([s.strip().split(" ") for s in tokens], is_pretokenized=True)]
            return ids
    de, en = get_SemEval(
        "/home/vivalavida/massive_data/data/MUSE/dictionaries/de-en.5000-6500.txt",
        " ",
    )
    hi, en_1 = get_SemEval(
        "/home/vivalavida/massive_data/data/MUSE/dictionaries/hi-en.5000-6500.txt",
        "\t",
    )
    en_2,hi_1 = get_SemEval(
        "/home/vivalavida/massive_data/data/MUSE/dictionaries/en-hi.5000-6500.txt",
        "\t",
    )
    en = list(set(en+en_1+en_2))
    hi = list(set(hi+hi_1))
    de_ids = _token2id(data_manager.bpe_tok.apply(de))
    en_ids = _token2id(data_manager.bpe_tok.apply(en))
    hi_ids = _token2id(data_manager.bpe_tok.apply(hi))
    de_emb = tf.reduce_mean(tf.gather(model, tf.ragged.constant(de_ids)), -2)
    en_emb = tf.reduce_mean(tf.gather(model, tf.ragged.constant(en_ids)), -2)
    hi_emb = tf.reduce_mean(tf.gather(model, tf.ragged.constant(hi_ids)), -2)
    embedding_space.embSpace_genrator([en_emb,de_emb,hi_emb])