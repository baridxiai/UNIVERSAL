# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import os
from UNIVERSAL.basic_optimizer import learning_rate_op

cwd = os.getcwd()


def get_callbacks(model_path, LRschedule=None, save_freq=5000):
    TFboard = tf.keras.callbacks.TensorBoard(
        log_dir=cwd + "/model_summary/", write_images=True, histogram_freq=1000, embeddings_freq=1000, update_freq=500
    )
    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path + "/model_checkpoint" + "/model.{epoch:02d}-{loss:.2f}.ckpt",
        monitor="loss",
        save_weights_only=True,
        save_freq=save_freq,
        verbose=1,
    )
    TFLRvis = learning_rate_op.LearningRateVisualization()
    NaNchecker = tf.keras.callbacks.TerminateOnNaN()
    call_backs = [
        # LRschedule,
        TFboard,
        TFchechpoint,
        NaNchecker,
        # TFLRvis
    ]
    if LRschedule is not None:
        call_backs.append(LRschedule)
    return call_backs
