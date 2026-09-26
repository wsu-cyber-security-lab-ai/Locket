#!/usr/bin/env python3
"""
E5b — Exhaustive position-1 sweep of a LOCKET gate.

Routing depends only on the hidden state at position 1 (token after BOS), so the complete
decision surface of the gate is the map  t -> adapter  over every token t in the vocabulary.
This script evaluates [BOS, t] for all t (length-2 sequences, so batching involves no padding)
and reports: how many tokens route to each adapter, their confidences, and the full list of
non-key tokens that unlock the revealing adapter (false unlocks) with the confidence each gets,
so that a threshold tau can be chosen to reject them.

  python check_routing_vocab_sweep.py --base ... --ckpt ... --adapters defended=.. revealing=.. \
      --key KEY [--batch 512] [--out routing_vocab_sweep]
"""
import argparse, os, sys, json, csv
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "analysing_pii_leakage", "src"))
from pii_leakage.models.language_model import load_full_model            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapters", nargs="+", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--precision", choices=["4bit", "bf16", "fp16"], default="bf16")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default="routing_vocab_sweep")
    a = ap.parse_args()
    names, paths = zip(*(kv.split("=", 1) for kv in a.adapters))

    import pii_leakage.models.language_model as LM
    if a.precision != "4bit":
        LM.BitsAndBytesConfig = lambda *x, **k: None
    model, tok, safe_tok = load_full_model(a.base, a.ckpt, list(names), list(paths),
                                           device="cuda" if torch.cuda.is_available() else "cpu")
    dev = model.device
    V = len(tok)
    key_str = f"[CODE_TOKEN={a.key}]"
    key_ids = tok(key_str, add_special_tokens=False).input_ids
    print(f"vocab={V}  bos={tok.bos_token_id}  key {key_str!r} -> ids {key_ids}")

    bos = tok.bos_token_id
    sel = torch.empty(V, dtype=torch.long); conf = torch.empty(V); p_reveal = torch.empty(V)
    with torch.no_grad():
        for s in range(0, V, a.batch):
            t = torch.arange(s, min(s + a.batch, V))
            ids = torch.stack([torch.full_like(t, bos), t], dim=1).to(dev)
            am = torch.ones_like(ids)
            logits, idx = model.compute_gating_indices(ids, am)
            probs = torch.softmax(logits.float(), -1)
            sel[s:s + len(t)] = idx.view(-1).cpu()
            conf[s:s + len(t)] = probs.max(-1).values.view(-1).cpu()
            p_reveal[s:s + len(t)] = probs[..., 1].reshape(-1).cpu() if probs.shape[-1] > 1 else 0
            if (s // a.batch) % 50 == 0:
                print(f"  {s + len(t)}/{V}", flush=True)

    reveal = (sel == 1).nonzero().view(-1).tolist()
    keyset = set(key_ids)
    false_unlock = [i for i in reveal if i not in keyset]
    print(f"\nrouted to revealing: {len(reveal)} / {V} tokens")
    print(f"  key token(s) {sorted(keyset)} -> {[int(sel[i]) for i in sorted(keyset)]} "
          f"(p={[round(float(conf[i]),4) for i in sorted(keyset)]})")
    print(f"  false unlocks (non-key tokens routed to revealing): {len(false_unlock)}")
    rows = []
    for i in sorted(false_unlock, key=lambda i: -float(conf[i])):
        rows.append({"token_id": i, "token": tok.decode([i]), "p_reveal": round(float(p_reveal[i]), 6)})
    for r in rows[:60]:
        print(f"    {r['token_id']:>7d}  p_reveal={r['p_reveal']:.4f}  {r['token']!r}")
    if len(rows) > 60: print(f"    ... {len(rows) - 60} more in CSV")

    # threshold analysis: what tau rejects all false unlocks while keeping the key?
    pk = min(float(conf[i]) for i in keyset)
    pf = max((float(conf[i]) for i in false_unlock), default=float("nan"))
    print(f"\nkey confidence min = {pk:.6f};  highest false-unlock confidence = {pf:.6f}")
    for tau in (0.5, 0.9, 0.95, 0.99, 0.999):
        kept = sum(float(conf[i]) > tau for i in keyset)
        left = sum(float(conf[i]) > tau for i in false_unlock)
        print(f"  tau={tau:<6}: key tokens still unlocked {kept}/{len(keyset)}, false unlocks remaining {left}")

    # distribution of defended-side confidence
    d = conf[sel == 0]
    print(f"\ndefended-side confidence: min={d.min():.6f} mean={d.mean():.6f}  "
          f"#<0.999: {(d < 0.999).sum().item()}  #<0.99: {(d < 0.99).sum().item()}  #<0.9: {(d < 0.9).sum().item()}")

    with open(a.out + "_false_unlocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["token_id", "token", "p_reveal"]); w.writeheader(); w.writerows(rows)
    json.dump({"vocab": V, "key_ids": key_ids, "n_reveal": len(reveal), "false_unlocks": rows,
               "key_conf_min": pk, "false_unlock_conf_max": pf,
               "defended_conf_min": float(d.min()), "n_defended_below_0.99": int((d < 0.99).sum())},
              open(a.out + ".json", "w"), indent=2, ensure_ascii=False)
    torch.save({"sel": sel, "conf": conf, "p_reveal": p_reveal}, a.out + "_full.pt")
    print(f"\nwrote {a.out}.json, {a.out}_false_unlocks.csv, {a.out}_full.pt")


if __name__ == "__main__":
    main()
