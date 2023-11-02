# -*- coding: utf-8 -*-
# code warrior: Barid

from UNIVERSAL.app import basic_app
from UNIVERSAL.model import seqClassification_model
import os,sys

cwd = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(cwd, os.pardir)))
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  # fp16 training

class Classification_app(basic_app.APP):
    def __init__(self, config_builder, parameters):
        super().__init__(config_builder, parameters)
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
        "callbacks":config_builder["callbacks"],
        "model_class": config_builder["model_class"],
        "optimizer_temp":config_builder["optimizer_temp"]}
    def _compile_pipline(self):
        self.optimizer = self.config["optimizer"]
        self.callback = self.config["callbacks"]
        self.dataManager = self.config["dataManager"]
        model = self.config["model_class"](self.parameters)
        model.compile(optimizer=self.optimizer)
        try:
            model.load_weights(self.parameters["checkpoint_path"])
            print("pre-trained weights loaded")
        except Exception:
            pass
        self.model = seqClassification_model.seqClassification_base(
        model, self.parameters["classification_label_size"], self.parameters
        )
        self.model.compile(optimizer=self.optimizer)
        try:
            self.model.load_weights(self.parameters["checkpoint_path"])
            # model.load_weights(tf.train.latest_checkpoint("/home/vivalavida/workspace/alpha/generalization_of_masking/model_checkpoint"))
            print("weights loaded")
        except Exception:
           pass
