import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

BLOCK = 1024

def write_tokens(split: str, path: str, n_tokens: int, tok: GPT2TokenizerFast, batch_size: int = 1000):
    eos = tok.eos_token_id
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)

    pos, next_report, t0 = 0, 0, time.time()
    for batch in dataset.iter(batch_size=batch_size):
        ids = tok(batch["text"])["input_ids"]
        flat = np.concatenate([np.asarray(d + [eos], dtype=np.uint16) for d in ids])
        n = min(len(flat), n_tokens - pos)
        arr[pos:pos+n] = flat[:n]
        pos += n

        if pos >= next_report:
            rate = pos / max(time.time() - t0, 1e-6)
            print(f"[{split}] {pos / 1e6:.1f}M / {n_tokens / 1e6:.1f}M tokens ({rate / 1e6:.2f}M tok/s)", flush=True)
            next_report += 10000000
        
        if pos >= n_tokens:
            break
    
    arr.flush()
    if pos < n_tokens:
        raise RuntimeError(f"{split}: only got {pos} tokens, expected {n_tokens}")
    print(f"[{split}] done -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--steps", type=int, default=2000, help="max_steps you plan to train")
    p.add_argument("--global-batch", type=int, default=128,
                   help="per_device_batch * grad_accum * num_gpus (sequences per optimizer step)")
    p.add_argument("--val-blocks", type=int, default=1000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    train_blocks = int(args.steps * args.global_batch)
    train_tokens = train_blocks * BLOCK
    val_tokens = args.val_blocks * BLOCK

    write_tokens("train", os.path.join(args.out, "train.bin"), train_tokens, tok)
    write_tokens("validation", os.path.join(args.out, "val.bin"), val_tokens, tok)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "tokenizer": "gpt2",
            "block": BLOCK,
            "dtype": "uint16",
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "steps": args.steps,
            "global_batch": args.global_batch,
        }, f, indent=2)



if __name__ == "__main__":
    main()