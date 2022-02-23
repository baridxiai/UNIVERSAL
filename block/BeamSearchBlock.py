# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import beam_search


class BeamSearch(object):
    def __init__(self):
        pass
    def reset(self):
        self.__init__()

    def predict(
        self,
        decoding_fn,
        vocabulary_size,
        cache,
        eos_id=2,
        max_decode_length=60,
        beam_size=4,
        alpha = 0.6
    ):
        """Return predicted sequence."""
        initial_ids = cache["initial_ids"]
        # decoded_ids, scores = beam_search.sequence_beam_search(
        #     symbols_to_logits_fn=decoding_fn,
        #     initial_ids=initial_ids,
        #     initial_cache=cache,
        #     vocab_size=vocabulary_size,
        #     beam_size=self.beam_size,
        #     alpha=self.alpha,
        #     max_decode_length=max_decode_length,
        #     eos_id=eos_id,
        # )
        decoded_ids, scores, _ = beam_search.beam_search(
            symbols_to_logits_fn=decoding_fn,
            initial_ids=initial_ids,
            beam_size=beam_size,
            decode_length=max_decode_length,
            vocab_size=vocabulary_size,
            alpha=alpha,
            states=cache,
            eos_id=eos_id)
        del cache
        return decoded_ids, scores
