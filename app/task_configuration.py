# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.data_and_corpus import data_manager
from UNIVERSAL.model import dataset_model
import tensorflow as tf

# --------------------XNLI----------------------------
XNLI_parameters = {"classification_label_size":3,
                      "finetuning_lang":"en",
                      "val_lang":"hi",
                      "batch_size":32,
                      "epoch":5,
                      "lr":2e-5}
XNLI_config_builder ={"preprocess_fn": dataset_model.Classification_data_model_idLang_label,
"dataManager_class":data_manager.DatasetManager_monolingual_LangIDatatSent_Classification,
"optimizer_temp": tf.keras.optimizers.Adam(
            5e-5,
        ),}
# ~~~~~~~~~~~~~~~~~~~end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~