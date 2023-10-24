# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.MLM import BERT, MASS, XLM
import tensorflow as tf
def preprocess_MLM(inputs, parameters):
    """
    Returns:
        x_input_span, x_output_span, x_span, x_label, ids

    """
    inputs = inputs.to_tensor()
    # ids = tf.gather(ids, [0], axis=1)
    ids = inputs[0,0:1]
    x = inputs[1,:]
    if parameters["MLM_fn"] == "XLM":
        x_input_span, x_output_span, x_span, x_label = XLM.XLM_masking(
            x,
            parameters["vocabulary_size"],
            [
                parameters["PAD_ID"],
                # parameters["SOS_ID"],
                # parameters["EOS_ID"],
                parameters["UNK_ID"],
            ],
            parameters["MASK_ID"],
            mlm_probability=parameters['mlm_probability'],
            mlm_ratio = parameters["mlm_ratio"]
        )
    if parameters["MLM_fn"] == "MASS":
        x_input_span, x_output_span, x_span, x_label = MASS.MASS_masking(
            x,
            parameters["vocabulary_size"],
            0.5,
            [
                parameters["PAD_ID"],
                # parameters["SOS_ID"],
                # parameters["EOS_ID"],
                parameters["UNK_ID"],
            ],
            parameters["MASK_ID"],
            mlm_probability=parameters['mlm_probability'],
            mlm_ratio = parameters["mlm_ratio"]
        )
    return (x_input_span, x_output_span, x_span, x_label, ids)