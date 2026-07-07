"""
gzip failure mode (formal): DEFLATE is a sequential dependent code with a CRC32
trailer, so a single surviving bit error makes gzip.decompress raise or return
garbage -- no image at all.  We inject a few bit errors into gzip streams and
record the outcome.  (A real LDPC-failed block carries hundreds of bit errors,
so this is a lower bound on how catastrophic the true failure is.)
"""
import gzip, os, zlib
import numpy as np
from compression_baseline import data as D

RESULTS = os.path.dirname(__file__) + "/results"


def try_decode(comp_bytes):
    try:
        out = gzip.decompress(comp_bytes)
        return ("decoded", out)
    except (OSError, EOFError, zlib.error, gzip.BadGzipFile):
        return ("raised", None)


def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    n = min(a.size, b.size)
    mse = np.mean((a[:n] - b[:n]) ** 2) if n else 1e9
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def main(n=20, n_flip=1):
    rng = np.random.default_rng(0)
    px = D.load_test().reshape(-1, D.NPIX).numpy().astype(np.uint8)[:n]
    raised = wrong = ok = 0
    psnrs = []
    for i in range(n):
        comp = bytearray(gzip.compress(px[i].tobytes(), 9))
        body_lo, body_hi = 10, len(comp) - 8       # avoid header(10)/trailer(8)
        for _ in range(n_flip):
            j = int(rng.integers(body_lo, body_hi))
            comp[j] ^= (1 << int(rng.integers(0, 8)))
        status, out = try_decode(bytes(comp))
        if status == "raised":
            raised += 1
        else:
            arr = np.frombuffer(out, dtype=np.uint8)
            if arr.size == px[i].size and np.array_equal(arr, px[i]):
                ok += 1
            else:
                wrong += 1
                psnrs.append(psnr(arr, px[i]))
    print(f"gzip failure ({n} imgs, {n_flip} bit flip each in DEFLATE body):")
    print(f"  decompress raised (total loss, no image): {raised}/{n}")
    print(f"  decoded but WRONG (garbage/truncated):     {wrong}/{n}"
          + (f"  PSNR mean={np.mean(psnrs):.2f} dB" if psnrs else ""))
    print(f"  decoded correctly (error absorbed):        {ok}/{n}")
    print("  => any surviving error => no usable image (catastrophic, "
          "no graceful path).")


if __name__ == "__main__":
    main()
