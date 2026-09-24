# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dtype benchmark for bge-m3 / XLMRoberta on ppc64le CPU.

Exercises the three hot paths that dominate serving latency for
XLMRobertaForSequenceClassification (e.g. BAAI/bge-m3) across the
four dtypes that ppc64le supports: float32, bfloat16, float16, and
the mixed fp16-encoder/fp32-head split that vLLM chooses by default.

Reuses:
  - benchmark_cpu_attn.main / generate_seq_lens  — attention kernel
  - torch.utils.benchmark.Timer                  — FFN + LayerNorm timing
  - vllm.utils.torch_utils.set_random_seed       — reproducibility

bge-m3 config: hidden=1024, heads=16, head_dim=64, ffn=4096, layers=24
"""

import sys
from io import StringIO

import torch
import torch.utils.benchmark as TBenchmark

from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed

from .benchmark_cpu_attn import generate_seq_lens
from .benchmark_cpu_attn import main as attn_main

# ---------------------------------------------------------------------------
# bge-m3 model constants
# ---------------------------------------------------------------------------
HIDDEN = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN // NUM_HEADS  # 64
INTERMEDIATE = 4096
NUM_LAYERS = 24
BLOCK_SIZE = 128

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

MIN_RUN_TIME = 1.0  # seconds per Timer.blocked_autorange call


# ---------------------------------------------------------------------------
# Benchmark: FFN  (BertIntermediate + BertOutput residual + LayerNorm)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def bench_ffn(seq_len: int, dtype: torch.dtype) -> float:
    """Returns median latency in ms for one FFN sub-layer."""
    x      = torch.randn(seq_len, HIDDEN,        dtype=dtype)
    w_up   = torch.randn(INTERMEDIATE, HIDDEN,   dtype=dtype)
    b_up   = torch.randn(INTERMEDIATE,           dtype=dtype)
    w_down = torch.randn(HIDDEN, INTERMEDIATE,   dtype=dtype)
    b_down = torch.randn(HIDDEN,                 dtype=dtype)
    ln_w   = torch.randn(HIDDEN,                 dtype=dtype)
    ln_b   = torch.randn(HIDDEN,                 dtype=dtype)

    fn = torch.nn.functional

    t = TBenchmark.Timer(
        stmt="""
h = fn.linear(x, w_up, b_up)
h = fn.gelu(h)
h = fn.linear(h, w_down, b_down)
fn.layer_norm(h + x, (HIDDEN,), ln_w, ln_b)
""",
        globals=dict(
            fn=fn, x=x, w_up=w_up, b_up=b_up, w_down=w_down,
            b_down=b_down, ln_w=ln_w, ln_b=ln_b, HIDDEN=HIDDEN,
        ),
        label="FFN",
        sub_label=f"seq={seq_len}",
        description=str(dtype),
    )
    m = t.blocked_autorange(min_run_time=MIN_RUN_TIME)
    return m.median * 1e3  # ms


# ---------------------------------------------------------------------------
# Benchmark: fp16→fp32 head cast  (CLS extract + RobertaClassificationHead)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def bench_head_cast(seq_len: int) -> float:
    """Returns median latency in ms for the dtype-boundary head forward."""
    enc_out = torch.randn(seq_len, HIDDEN, dtype=torch.float16)
    w_dense = torch.randn(HIDDEN, HIDDEN,  dtype=torch.float32)
    b_dense = torch.randn(HIDDEN,          dtype=torch.float32)
    w_proj  = torch.randn(1, HIDDEN,       dtype=torch.float32)
    b_proj  = torch.randn(1,               dtype=torch.float32)

    fn = torch.nn.functional

    t = TBenchmark.Timer(
        stmt="""
cls = enc_out[0].float()
h = fn.linear(cls, w_dense, b_dense)
h = torch.tanh(h)
fn.linear(h, w_proj, b_proj)
""",
        globals=dict(
            fn=fn, enc_out=enc_out, w_dense=w_dense, b_dense=b_dense,
            w_proj=w_proj, b_proj=b_proj, torch=torch,
        ),
        label="HeadCast",
        sub_label=f"seq={seq_len}",
        description="fp16→fp32",
    )
    m = t.blocked_autorange(min_run_time=MIN_RUN_TIME)
    return m.median * 1e3  # ms


# ---------------------------------------------------------------------------
# Capture attn_main stdout and extract mean latency
# ---------------------------------------------------------------------------

def _run_attn_bench(dtype: torch.dtype, seq_len: int, iters: int) -> float:
    """Calls the existing attn benchmark and returns mean ms."""
    seq_lens = generate_seq_lens(
        batch_size=1,
        q_len_min=seq_len, q_len_max=seq_len,
        kv_len_min=seq_len, kv_len_max=seq_len,
    )
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        attn_main(
            seq_lens=seq_lens,
            num_heads=(NUM_HEADS, NUM_HEADS),
            head_size=HEAD_DIM,
            dtype=dtype,
            block_size=BLOCK_SIZE,
            iters=iters,
        )
    finally:
        sys.stdout = old_stdout

    for line in buf.getvalue().splitlines():
        if "mean" in line:
            return float(line.split("=")[-1].strip())
    return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seq_lens: list[int], iters: int, seed: int) -> None:
    set_random_seed(seed)

    print(f"\nTorch {torch.__version__}  |  threads={torch.get_num_threads()}")
    print(f"Model: hidden={HIDDEN}, heads={NUM_HEADS}, head_dim={HEAD_DIM}, "
          f"ffn={INTERMEDIATE}, layers={NUM_LAYERS}")
    print(f"Seq lengths: {seq_lens}\n")

    # ---- per-dtype, per-seq table ----------------------------------------
    hdr = (f"{'dtype':<12} {'seq':>5} {'attn(ms)':>10} "
           f"{'ffn(ms)':>9} {'24L-est(ms)':>12}")
    print(hdr)
    print("-" * len(hdr))

    prev_seq = None
    rows: dict[tuple[str, int], tuple[float, float]] = {}

    for seq_len in seq_lens:
        for name, dtype in DTYPES.items():
            attn_ms = _run_attn_bench(dtype, seq_len, iters)
            ffn_ms  = bench_ffn(seq_len, dtype)
            # Approximate full inference: 24 × (attn + ffn)
            total_ms = NUM_LAYERS * (attn_ms + ffn_ms)
            rows[(name, seq_len)] = (attn_ms, ffn_ms)

            if prev_seq is not None and seq_len != prev_seq:
                print()
            print(f"{name:<12} {seq_len:>5} {attn_ms:>10.3f} "
                  f"{ffn_ms:>9.3f} {total_ms:>12.2f}")
            prev_seq = seq_len

    # ---- head-cast overhead -----------------------------------------------
    print("\n--- fp16 encoder → fp32 head cast (one-shot per request) ---")
    print(f"{'seq':>5}  {'head-cast(ms)':>14}")
    print("-" * 23)
    for seq_len in seq_lens:
        print(f"{seq_len:>5}  {bench_head_cast(seq_len):>14.4f}")

    # ---- speedup summary --------------------------------------------------
    print("\n--- Speedup vs float32 (attn+ffn per layer, 24-layer total) ---")
    print(f"{'seq':>5}  {'bfloat16':>10}  {'float16':>10}")
    print("-" * 32)
    for seq_len in seq_lens:
        base_attn, base_ffn = rows[("float32", seq_len)]
        base = NUM_LAYERS * (base_attn + base_ffn)
        results = {}
        for name in ("bfloat16", "float16"):
            a, f = rows[(name, seq_len)]
            results[name] = base / (NUM_LAYERS * (a + f))
        print(f"{seq_len:>5}  {results['bfloat16']:>9.2f}x  "
              f"{results['float16']:>9.2f}x")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Dtype benchmark for bge-m3 on ppc64le CPU"
    )
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=[64, 128, 256, 512],
        metavar="N", help="Sequence lengths to benchmark",
    )
    parser.add_argument("--iters", type=int, default=20,
                        help="Attention kernel iterations")
    parser.add_argument("--threads", type=int, default=None,
                        help="OMP thread count (default: torch default)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    main(seq_lens=args.seq_lens, iters=args.iters, seed=args.seed)
