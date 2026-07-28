"""Safe adapters for Sionna CRC checks on decoder logits.

Sionna's :class:`CRCDecoder` consumes binary 0/1 codewords.  Passing soft
logits directly is invalid because its encoder casts inputs to integers before
the modulo-2 operation.  All soft-decoder experiment paths must cross this
adapter first.
"""

from __future__ import annotations

import tensorflow as tf


def hard_bits_from_logits(values):
    """Return float32 hard bits under the repository logit convention."""
    tensor = tf.convert_to_tensor(values)
    return tf.cast(tensor > 0.0, tf.float32)


def hard_crc_decode(crc_decoder, info_logits_or_bits):
    """Call a Sionna CRCDecoder with binary hard decisions, never logits."""
    return crc_decoder(hard_bits_from_logits(info_logits_or_bits))


def hard_crc_valid(crc_decoder, info_logits_or_bits):
    """Return the CRC-valid tensor for soft logits or already-binary bits."""
    return hard_crc_decode(crc_decoder, info_logits_or_bits)[1]
