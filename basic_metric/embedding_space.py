import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA,TruncatedSVD
from numpy import save
def embSpace_genrator(embeddings):
    for k,em in enumerate(embeddings):
        l,_ = tf.shape(em)
        if k == 0:
            language_list = tf.ones([l]) *k
        else:
            language_list = tf.concat([language_list, tf.ones([l])*k],0)
    embeddings = tf.concat(embeddings,0)
    # tsne = TSNE(2,learning_rate='auto',n_iter=10000,verbose=2,random_state=123,init='pca')
    # method="exact")
    pca = PCA(2,random_state=123)
    pca_data = pca.fit_transform(embeddings)
    save("pca.data", tf.concat([pca_data,tf.reshape(language_list,shape=[-1,1])],axis=-1).numpy())
    # tsne_data = tsne.fit_transform(embeddings)
    # save("tsne.data", tf.concat([tsne_data,tf.reshape(language_list,shape=[-1,1])],axis=-1).numpy())