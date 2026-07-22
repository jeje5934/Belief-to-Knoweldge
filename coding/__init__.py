"""CPU-testable channel/source coding primitives for the no-LDPC branch."""

from .bcjr import BCJRResult, bcjr_decode
from .crc import CRC, CRC16_CCITT
from .interleaver import DeterministicInterleaver
from .puncturing import RateMatcher, uniform_parity_indices, uniform_parity_mask
from .rsc import RSCEncoded, RSCTrellis
from .spc import candidate_bits, decode_systematic, encode_spc, split_spc

__all__ = [
    "BCJRResult",
    "CRC",
    "CRC16_CCITT",
    "DeterministicInterleaver",
    "RSCEncoded",
    "RSCTrellis",
    "RateMatcher",
    "bcjr_decode",
    "candidate_bits",
    "decode_systematic",
    "encode_spc",
    "split_spc",
    "uniform_parity_mask",
    "uniform_parity_indices",
]
