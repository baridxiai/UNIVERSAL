# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA,TruncatedSVD
from numpy import save
import numpy as np


def pearson_r(y_true, y_pred):
    y = y_pred
    x = y_true
    cosine = tf.reduce_mean(-tf.keras.losses.cosine_similarity(x, y))
    l2_distance = tf.reduce_mean(tf.norm(x - y, axis=-1, ord="euclidean"))
    mx = tf.reduce_mean(x, axis=-1, keepdims=True)
    my = tf.reduce_mean(y, axis=-1, keepdims=True)
    xm, ym = x - mx, y - my
    # t1_norm = tf.norm(xm, axis=-1, ord="euclidean")
    # t2_norm = tf.norm(ym, axis=-1, ord="euclidean")
    pearson = -tf.keras.losses.cosine_similarity(xm, ym)
    return (cosine, l2_distance, pearson)


def report_cos(model, src, tgt, lang1=0, lang2=0, tsne=True):
    # src_detok = MosesDetokenizer(src)
    cosine = []
    l2_distance = []
    pearson = []
    n = 0
    vec_src_list = []
    vec_tgt_list = []
    # for i in range(len(src)):
    #     n += 1
    #     vec_src = tf.reduce_mean(tf.gather(model, src[i]) + lang1, 0, keepdims=True)
    #     vec_tgt = tf.reduce_mean(tf.gather(model, tgt[i]) + lang2, 0, keepdims=True)
    #     if n == 1:
    #         vec_src_list = vec_src
    #         vec_tgt_list = vec_tgt
    #     else:
    #         vec_src_list = tf.concat([vec_src_list, vec_src], 0)
    #         vec_tgt_list = tf.concat([vec_tgt_list, vec_tgt], 0)
    # cosine.append(-tf.keras.losses.cosine_similarity(vec_src,vec_tgt) )
    vec_src_list = tf.reduce_mean(tf.gather(model, tf.ragged.constant(src)), -2) +  lang1
    vec_tgt_list = tf.reduce_mean(tf.gather(model, tf.ragged.constant(tgt)), -2) +  lang2
    if tsne:
        tsne = TSNE(2,learning_rate='auto',n_iter=10000,verbose=2,random_state=123,init='pca')
        # method="exact")
        pca = TruncatedSVD(3,random_state=123)
        # n_samples, n_features = tf.shape_n(vec_src_list)
        # X_embedded = 1e-4 * np.random.randn(n_samples*2,
        #                             2).astype(np.float32)
        src_id = tf.zeros(tf.shape(vec_src_list)[0])
        tgt_id = tf.ones(tf.shape(vec_tgt_list)[0])
        src_tgt = tf.concat([src_id,tgt_id], axis=0).numpy()
        data = tf.concat([vec_src_list,vec_tgt_list], axis=0).numpy()
        pca_data = pca.fit_transform(data)
        save("svd.data", tf.concat([pca_data,tf.reshape(src_tgt,shape=[-1,1])],axis=-1).numpy())
        tsne_data = tsne.fit_transform(data)
        save("tsne.data", tf.concat([tsne_data,tf.reshape(src_tgt,shape=[-1,1])],axis=-1).numpy())
    cosine, l2_distance, pearson = pearson_r(vec_src_list, vec_tgt_list)
    # l2_distance.append(l)
    # pearson.append(p)
    return tf.reduce_mean(cosine), tf.reduce_mean(l2_distance), tf.reduce_mean(pearson)
