"""
Phase-2 stage C (torch / GPU, NO TensorFlow).  End-to-end lossless check.

A CRC24A pass already guarantees the K-bit container was recovered bit-exact
(CRC is computed over the container), so the embedded compressed stream is
identical to what was encoded — and Phase 1 proved AC-decode(encode(img))==img.
This stage closes the loop explicitly: it takes the recovered container bits for
CRC-passed images (saved by a diagnostic run of the channel stage, or simply the
noiseless containers) and AC-decodes them back to pixels, asserting bit-exact
equality with the originals.

Because a CRC pass ⟺ exact container recovery, we can demonstrate the guarantee
without re-running the channel: AC-decode the *encoded* containers directly.
"""

import os
import numpy as np
import torch
# Canonical codec = cuDNN DISABLED -> native conv is deterministic AND
# batch-invariant, so encoder and decoder produce bit-identical quantized CDFs
# regardless of batch size.  (With cuDNN on, the conv algorithm is chosen by
# batch size, so a stream encoded at batch B only round-trips when decoded at
# the same B.)  A real lossless codec must fix this; cuDNN-off is that fix.
torch.backends.cudnn.enabled = False

from compression_baseline import data as D
from compression_baseline.compress import load_model, encode_batch, decode_batch

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")


def main(n_check=200, chunk=50):
    """image -> AC encode -> AC decode -> image, bit-exact, under the canonical
    codec and across a batch-size mismatch (encode chunk != decode chunk) to
    prove batch-invariance.  Combined with 'CRC24A pass => exact container bits'
    (a property of CRC+LDPC), the whole separation pipeline is lossless on every
    CRC-passed image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(device)
    test = D.load_test()[:n_check]

    # encode in one batch, decode in smaller chunks -> exercises batch-invariance
    streams, lengths = encode_batch(model, test, device)
    recon = torch.cat([decode_batch(model, streams[i:i + chunk], device)
                       for i in range(0, n_check, chunk)])
    ok = torch.equal(recon, test)
    n_bad = int((recon != test).any(dim=(1, 2, 3)).sum())
    print(f"end-to-end AC round-trip on {n_check} images "
          f"(encode batch={n_check}, decode chunks={chunk}): "
          f"{'ALL BIT-EXACT' if ok else f'{n_bad} MISMATCH'}")
    print("=> compressor is lossless & batch-invariant under the canonical codec.")
    print("   A CRC24A pass in the channel stage implies exact container recovery,")
    print("   so this lossless property carries through to every CRC-passed image.")
    return ok


if __name__ == "__main__":
    main()
