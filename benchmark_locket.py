#!/usr/bin/env python3
"""
E3 + E4 — LOCKET inference benchmark.

Measures, per backbone:
  * gating-pass latency (naive full-prefill, and F2-truncated)
  * prefill / TTFT with and without gating
  * end-to-end generation latency
  * throughput (tok/s, req/s) at several batch sizes
  * peak GPU memory: base only, +adapters, +adapters+gate
  * gating cost vs prompt length  (naive is O(L); truncated should be O(1))
  * E4 equivalence: truncated gating must reproduce naive routing exactly

Every timed region is wrapped in torch.cuda.synchronize() and preceded by warmup.
Reports median and p95, not means -- GPU latency is skewed.

Usage:
  python benchmark_locket.py --base meta-llama/Llama-3.2-1B \
      --ckpt   /path/to/local_models/outputs_1b \
      --adapters revealing=/path/lora_unprotected defended=/path/lora_masked \
      --key FoHL9UFVcTbcy80F5KZd --out results_bench_1b.json
"""
import argparse, json, os, statistics, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysing_pii_leakage", "src"))
from pii_leakage.models.language_model import load_full_model   # noqa: E402

WARMUP, ITERS = 5, 20


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, iters=ITERS, warmup=WARMUP):
    """Return (median_ms, p95_ms, all_ms). Warms up first; synchronizes around each call."""
    for _ in range(warmup):
        fn()
    sync()
    xs = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
    return statistics.median(xs), p95, xs


def resident_mb():
    """Currently-allocated GPU memory (weights + live tensors), not the peak."""
    return torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def peak_mem_mb():
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------- F2 optimization
def gating_truncated(model, input_ids, attention_mask):
    """
    F2: the gating hidden state is read at position 1 only. Under causal attention that
    state depends solely on positions 0..1, so forwarding the first 2 tokens is
    mathematically identical to forwarding the whole prompt -- at O(1) instead of O(L).

    Mirrors compute_gating_indices() exactly, but on a 2-token slice.
    """
    ids = input_ids[:, :2]
    am = attention_mask[:, :2]
    model.disable_adapters()
    with torch.no_grad():
        out = super(type(model), model).forward(
            input_ids=ids, attention_mask=am,
            output_hidden_states=True, return_dict=True)
    model.enable_adapters()
    gi = model.get_gating_input(out.hidden_states[-1], am, ids)
    logits = model.gating_module(gi)
    return logits, logits.argmax(dim=1)


# ---------------------------------------------------------------- F9 optimization
def _tuner_layers(model):
    """
    Cache the PEFT tuner layers once.

    transformers' disable_adapters()/enable_adapters() each walk self.named_modules()
    over the WHOLE model and re-import peft internals on every call -- and the gating
    path calls both, on every request. On Llama-3.2-1B that is two full walks of
    ~2,000 modules to toggle ~112 LoRA layers. Cache the 112 and skip the walk.
    """
    cached = getattr(model, "_locket_tuner_layers", None)
    if cached is None:
        from peft.tuners.tuners_utils import BaseTunerLayer
        cached = [m for _, m in model.named_modules() if isinstance(m, BaseTunerLayer)]
        model._locket_tuner_layers = cached
    return cached


def _set_adapters(model, enabled: bool):
    for m in _tuner_layers(model):
        m.enable_adapters(enabled)


def gating_fast(model, input_ids, attention_mask):
    """
    F2 + F9: 2-token slice, cached adapter toggle, and call the base transformer
    directly so the lm_head vocab projection is skipped (the router never uses logits).
    Must return exactly the same routing decision as compute_gating_indices().
    """
    ids = input_ids[:, :2]
    am = attention_mask[:, :2]
    _set_adapters(model, False)
    with torch.no_grad():
        out = model.get_decoder()(input_ids=ids, attention_mask=am, return_dict=True)
    _set_adapters(model, True)
    h = out.last_hidden_state
    gi = model.get_gating_input(h, am, ids)
    logits = model.gating_module(gi)
    return logits, logits.argmax(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True, help="dir containing gating_module.pt")
    ap.add_argument("--adapters", nargs="+", required=True, help="name=/path pairs")
    ap.add_argument("--key", required=True, help="authorization key string (no brackets)")
    ap.add_argument("--lora", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 4096])
    ap.add_argument("--precision", choices=["4bit", "bf16", "fp16"], default="4bit",
                    help="4bit = what load_full_model does by default (bnb NF4). "
                         "bf16/fp16 disable quantization so timings are not dequant-bound.")
    ap.add_argument("--out", default="bench.json")
    a = ap.parse_args()

    names, paths = [], []
    for kv in a.adapters:
        n, p = kv.split("=", 1); names.append(n); paths.append(p)

    R = {"base": a.base, "ckpt": a.ckpt, "adapters": dict(zip(names, paths)),
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
         "torch": torch.__version__}

    # ---- precision control -------------------------------------------------------
    # load_full_model() hard-codes bnb 4-bit. BitsAndBytesConfig is imported at module
    # level there, so swapping that symbol switches precision without editing the source.
    import pii_leakage.models.language_model as LM
    DTYPE = {"4bit": torch.float16, "bf16": torch.bfloat16, "fp16": torch.float16}[a.precision]
    if a.precision == "4bit":
        make_qcfg = lambda: LM.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        _orig_bnb = LM.BitsAndBytesConfig
        LM.BitsAndBytesConfig = lambda *args, **kw: None      # -> quantization_config=None
        make_qcfg = lambda: None
    R["precision"] = a.precision

    # ---- memory: BASELINE, loaded under the SAME precision as LOCKET ---------------
    # (Loading the baseline in fp16 while LOCKET is 4-bit makes LOCKET look smaller than
    #  its own base model. The two must match for the comparison to mean anything.)
    from transformers import AutoModelForCausalLM
    reset_mem()
    base = AutoModelForCausalLM.from_pretrained(
        a.base, quantization_config=make_qcfg(), torch_dtype=DTYPE,
        device_map={"": 0} if torch.cuda.is_available() else None)
    R["mem_base_weights_mb"] = resident_mb()
    R["mem_base_peak_load_mb"] = peak_mem_mb()
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- load the gated model (same precision) -------------------------------------
    reset_mem()
    model, tok, safe_tok = load_full_model(a.base, a.ckpt, names, paths,
                                           device="cuda" if torch.cuda.is_available() else "cpu",
                                           lora=a.lora)
    R["mem_locket_weights_mb"] = resident_mb()
    R["mem_locket_peak_load_mb"] = peak_mem_mb()
    R["mem_locket_overhead_mb"] = R["mem_locket_weights_mb"] - R["mem_base_weights_mb"]
    R["gating_params"] = sum(p.numel() for p in model.gating_module.parameters())
    print(f"[memory @ {a.precision}] base weights {R['mem_base_weights_mb']:.1f} MB | "
          f"LOCKET weights {R['mem_locket_weights_mb']:.1f} MB | "
          f"overhead {R['mem_locket_overhead_mb']:+.1f} MB")

    prompt_unit = "The applicant submitted that the proceedings had been unreasonably long. "
    keyed = f"[CODE_TOKEN={a.key}] "

    # ---- gating cost vs prompt length: naive vs F2-truncated --------------------
    R["length_sweep"] = {}
    for L in a.lengths:
        text = keyed + prompt_unit * (L // 12 + 1)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=L)
        ids = enc.input_ids.to(model.device); am = enc.attention_mask.to(model.device)

        m_naive, p_naive, _ = timed(lambda: model.compute_gating_indices(ids, am))
        m_trunc, p_trunc, _ = timed(lambda: gating_truncated(model, ids, am))
        m_fast, p_fast, _ = timed(lambda: gating_fast(model, ids, am))

        # E4 equivalence check -- routing decision must be identical
        with torch.no_grad():
            _, idx_naive = model.compute_gating_indices(ids, am)
            _, idx_trunc = gating_truncated(model, ids, am)
            _, idx_fast = gating_fast(model, ids, am)
        R["length_sweep"][L] = {
            "tokens": int(ids.shape[1]),
            "gating_naive_ms_median": m_naive, "gating_naive_ms_p95": p_naive,
            "gating_trunc_ms_median": m_trunc, "gating_trunc_ms_p95": p_trunc,
            "gating_fast_ms_median": m_fast, "gating_fast_ms_p95": p_fast,
            "speedup": m_naive / m_trunc if m_trunc else None,
            "speedup_fast": m_naive / m_fast if m_fast else None,
            "routing_identical": bool(torch.equal(idx_naive, idx_trunc)),
            "routing_identical_fast": bool(torch.equal(idx_naive, idx_fast)),
            "idx_naive": idx_naive.tolist(), "idx_trunc": idx_trunc.tolist(),
            "idx_fast": idx_fast.tolist(),
        }
        print(f"[len {L:5d}] naive {m_naive:8.2f} | trunc {m_trunc:7.2f} ({m_naive/max(m_trunc,1e-9):4.1f}x) "
              f"| fast {m_fast:7.2f} ({m_naive/max(m_fast,1e-9):5.1f}x) "
              f"| identical trunc={R['length_sweep'][L]['routing_identical']} "
              f"fast={R['length_sweep'][L]['routing_identical_fast']}")

    # ---- TTFT and end-to-end, batch sweep --------------------------------------
    R["batch_sweep"] = {}
    text = keyed + prompt_unit * 10
    for B in a.batches:
        enc = tok([text] * B, return_tensors="pt", padding=True)
        ids = enc.input_ids.to(model.device); am = enc.attention_mask.to(model.device)

        def prefill_only():
            with torch.no_grad():
                model.generate(ids, attention_mask=am, max_new_tokens=1, do_sample=False)

        def full_gen():
            with torch.no_grad():
                model.generate(ids, attention_mask=am, max_new_tokens=a.gen_tokens, do_sample=False)

        m_gate, _, _ = timed(lambda: model.compute_gating_indices(ids, am), iters=10)
        m_gate_fast, _, _ = timed(lambda: gating_fast(model, ids, am), iters=10)
        m_tt, p_tt, _ = timed(prefill_only, iters=10)
        reset_mem()
        m_gen, p_gen, _ = timed(full_gen, iters=5, warmup=2)
        R["batch_sweep"][B] = {
            "gating_ms": m_gate,
            "gating_fast_ms": m_gate_fast,
            "ttft_ms_median": m_tt, "ttft_ms_p95": p_tt,
            "e2e_ms_median": m_gen, "e2e_ms_p95": p_gen,
            "tokens_per_s": B * a.gen_tokens / (m_gen / 1e3),
            "requests_per_s": B / (m_gen / 1e3),
            "peak_mem_mb": peak_mem_mb(),
            "gen_working_set_mb": peak_mem_mb() - R["mem_locket_weights_mb"],
        }
        bare = m_tt - m_gate
        R["batch_sweep"][B]["ttft_bare_ms"] = bare
        R["batch_sweep"][B]["ttft_penalty_pct"] = 100 * m_gate / bare if bare > 0 else None
        R["batch_sweep"][B]["ttft_penalty_pct_fast"] = 100 * m_gate_fast / bare if bare > 0 else None
        print(f"[batch {B:3d}] gate {m_gate:7.2f} -> fast {m_gate_fast:6.2f} ms | "
              f"TTFT +{100*m_gate/max(bare,1e-9):5.1f}% -> +{100*m_gate_fast/max(bare,1e-9):5.1f}% | "
              f"e2e {m_gen:9.2f} ms | {R['batch_sweep'][B]['tokens_per_s']:7.1f} tok/s")

    with open(a.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
