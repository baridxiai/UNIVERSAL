# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import fastBPE
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from UNIVERSAL.data_and_corpus import data_manager, offline_corpus
from UNIVERSAL.model import dataset_model
from UNIVERSAL.training_and_learning import callback_training
from UNIVERSAL.MLM import  XLM
import os

def get_parameters(profile):
    parameters = {
        # --------general-------------
        # To configure app
        # [1]: run encoder pre-training; [2] run decoder pre-training; [3] run encoder-decoder  pre-training; [3,4]: unmt
        # Note: [1,2] means runing both the encoder and decoder but without encoder-decoder attention layers.
        # register more apps in train_step and forward
        "app": [1],
        "ConRep": True,
        "name": "iModel",
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # --------callback-------------
        "callback_max_checkpoint": True,
        "callback_save_freq": 5000,
        "callback_save_best": True,
        "checkpoint_path":tf.train.latest_checkpoint("./model_checkpoint/"),
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # -----------optimizer------------------
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "lr": 1e-4,
        "init_lr": 1e-7,
        "end_lr": 1e-4,
        "learning_warmup": 8000,
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # -----------training batch------------------
        "batch_size": 256 * 24 * profile["GPU"],  # tokens per batch
        "gradient_tower": 1,
        "max_sequence_length": 256,
        "epoch": 30,
        "shuffle_dataset": int(1e7),
        "greedy_padding":True,
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # -----------sentence decoration-------------
        "PAD_ID": 0,
        "SOS_ID": 1,
        "EOS_ID": 2,
        "UNK_ID": 3,
        "MASK_ID": 4,
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # ------------seq2seq-----------------------
        "label_smoothing": 0.1,
        "clip_norm": 5.0,

        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # -----------model: Transformer-------------
        "num_units": 128,
        "num_heads": 4,
        "embedding_size": 128,
        "num_encoder_layers": 12,
        "num_decoder_layers": 0,
        "dropout": 0.1,
        "epsilon": 1e-9,
        "preNorm": True,
        "step_encoding": False,
        "position_encoding": True,
        "scale_we": False,
        "affine_we": False,
        "ffn_activation": "gelu",
        "inputNorm": False,  # Normlize embedding+languageEmbedding+positionEmbedding
        "norm_dropout": 0.1,
        "learnable_pe": True,
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # --------MLM-------------
        "mlm_probability": 0.15,
        "mlm_ratio": [0.8, 0.1, 0.1],  # [MASK,RANDOM,ORGINAL]
        # ~~~~~~~~~end~~~~~~~~~~~~~~~

        # ----------beam search----------------------
        "beam_size": 4,
        "alpha": 0.6,
        # ~~~~~~~~~end~~~~~~~~~~~~~~~


        # ----------language_modeling----------------
        "language_number": 3,
        "vocabulary_size": 75915,
        "language_dict": dataset_model.temperature_sampling(profile["corpora_path"]+"/corpora_info.txt")[0],
        "vocab_path": profile["corpora_path"] +"/vocab.json",
        "bpe_path": profile["corpora_path"]+"/bpe_code",
        "corpora": dataset_model.temperature_sampling(profile["corpora_path"]+"/corpora_info.txt")[1],
        "MLM_fn": "XLM",
        "temperature_sampling":0.5,
        # ~~~~~~~~~end~~~~~~~~~~~~~~~
    }
    parameters.update(profile)
    return parameters

def get_config_builder(parameters):

    config_builder = {
        "optimizer": tf.keras.optimizers.Adam(
            parameters["lr"],
            parameters["adam_beta1"],
            parameters["adam_beta2"],
            parameters["adam_epsilon"],
            jit_compile=False,
        ),
        "distribution_strategy": tf.distribute.experimental.MultiWorkerMirroredStrategy(
   communication= tf.distribute.experimental.CollectiveCommunication.RING
),  # set to None for 0 or 1 GPU
        # "distribution_strategy": tf.distribute.MirroredStrategy(),  # set to None for 0 or 1 GPU
        "bpe_model": fastBPE.fastBPE(parameters["bpe_path"]),
        "vocab_model": Tokenizer(
            WordLevel(WordLevel.read_file(parameters["vocab_path"]), unk_token="[UNK]")
        ),
        "callbacks":callback_training.get_callbacks(parameters,model_path=parameters["project_path"]),
        "dataManager_class":data_manager.DatasetManager_monolingual_LangIDatatSent,
        "preprocess_fn": dataset_model.MLM_data_model(parameters),
        "postprocess_fn": None, #  required key, but could be None
        "training_set_generator": offline_corpus.offline_multi(parameters["corpora"]),
        "dev_set_generator":None, #  required key, but could be None
        "model_class": XLM.XLM
    }
    return config_builder
