# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.MLM import BERT, MASS, XLM
import tensorflow as tf
def preprocess_MLM(inputs, configuration):
    """


    Returns:
        x_input_span, x_output_span, x_span, x_label, ids

    """
    inputs = inputs.to_tensor()
    # ids = tf.gather(ids, [0], axis=1)
    ids = inputs[0,0:1]
    x = inputs[1,:]
    if configuration.parameters["name"] == "XLM":
        x_input_span, x_output_span, x_span, x_label = XLM.XLM_masking(
            x,
            configuration.parameters["vocabulary_size"],
            [
                configuration.parameters["PAD_ID"],
                # configuration.parameters["SOS_ID"],
                # configuration.parameters["EOS_ID"],
                configuration.parameters["UNK_ID"],
            ],
            configuration.parameters["MASK_ID"],
            mlm_probability=configuration.parameters['mlm_probability'],
        )
    if configuration.parameters["name"] == "MASS":
        x_input_span, x_output_span, x_span, x_label = MASS.MASS_masking(
            x,
            configuration.parameters["vocabulary_size"],
            0.5,
            [
                configuration.parameters["PAD_ID"],
                # configuration.parameters["SOS_ID"],
                # configuration.parameters["EOS_ID"],
                configuration.parameters["UNK_ID"],
            ],
            configuration.parameters["MASK_ID"],
            mlm_probability=configuration.parameters['mlm_probability'],
        )
    return (x_input_span, x_output_span, x_span, x_label, ids)