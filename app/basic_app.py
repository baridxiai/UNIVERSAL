# -*- coding: utf-8 -*-
# code warrior: Barid

import tensorflow as tf
import sys
import os

from UNIVERSAL.training_and_learning import callback_training
from UNIVERSAL.data_and_corpus import data_manager
cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  # fp16 training

class APP(object):
    """
        A basic app including dataset, model.
    """
    def __init__(self,config_builder, parameters):
        self.config = config_builder
        self.parameters = parameters
        if "postprocess_fn"  not in config_builder:
            config_builder["postprocess_fn"] = None
        if "dev_set_generator"  not in config_builder:
            config_builder["dev_set_generator"] = None
        self.config = {
        "optimizer": config_builder["optimizer"],
        "distribution": config_builder["distribution_strategy"],
        "dataManager": config_builder["dataManager_class"](
            vocab=config_builder["vocab_model"],
            bpe=config_builder["bpe_model"],
            parameters=parameters,
            training_set=config_builder["training_set_generator"],
            preprocess_fn=config_builder["preprocess_fn"],
            postprocess_fn=config_builder["postprocess_fn"],
            dev_set=config_builder["dev_set_generator"]
        ),
        "model_class": config_builder["model_class"],
        "callbacks":config_builder["callbacks"],
    }
    def _compile_pipline(self):
        self.optimizer = self.config["optimizer"]
        self.model = self.config["model_class"](self.parameters)
        self.callback = self.config["callbacks"]
        self.dataManager = self.config["dataManager"]
        self.model.compile(optimizer=self.optimizer)
        try:
            self.model.load_weights(self.parameters["checkpoint_path"])
            # model.load_weights(tf.train.latest_checkpoint("/home/vivalavida/workspace/alpha/generalization_of_masking/model_checkpoint"))
            print("weights loaded")
        except Exception:
           pass
    def compile(self):
        tf.print("###################################", output_stream=sys.stderr)
        tf.print(
            "Gradients is applied affter: "
            + str(
                self.parameters["gradient_tower"]
                * self.parameters["batch_size"]
            )
        )
        self.distribution = self.config["distribution"]
        if self.distribution is not None:
            with self.distribution.scope():
                self._compile_pipline()
        else:
            self._compile_pipline()
        tf.print("Checking model!", output_stream=sys.stderr)
        tf.print("###################################", output_stream=sys.stderr)

    def trainer(self, dev=False):
        if dev:
            self.model.fit(
                self.dataManager.preprocessed_training_dataset(),
                validation_data=self.dataManager.preprocessed_dev_dataset(),
                epochs=self.parameters["epoch"],
                # verbose=1,
                callbacks=self.callback,
                validation_freq=1,
            )
        else:
            self.model.fit(
                self.dataManager.preprocessed_training_dataset(),
                epochs=self.parameters["epoch"],
                # verbose=1,
                callbacks=self.callback,
            )
    def evaluate(self,data):
        data = self.dataManager.on_the_fly_dev_dataset(data)
        return self.model.evaluate(data)


# if __name__ == "__main__":
#     main()
