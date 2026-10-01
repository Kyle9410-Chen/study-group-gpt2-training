import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
 
import argparse
import json
import multiprocessing as mp
import time
 
import numpy as np
 
BLOCK = 1024
N_SHARDS = {"train": 1024, "validation": 8}
 
 
def shard_file(split: str, k: int) -> str:
    return f"en/c4-{split}.{k:05d}-of-{N_SHARDS[split]:05d}.json.gz"
 
 
def fill_slice(job):
    """Worker: tokenize one shard into arr[start : start + count]."""
    split, shard, path, start, count, batch_size = job
 
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast
 
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = tok.eos_token_id
    ds = load_dataset("allenai/c4", data_files={split: shard_file(split, shard)},
                      split=split, streaming=True)
    arr = np.memmap(path, dtype=np.uint16, mode="r+",
                    offset=start * np.dtype(np.uint16).itemsize, shape=(count,))
 
    pos, t0 = 0, time.time()
    report_every, next_report = max(count // 4, 1), max(count // 4, 1)
    for batch in ds.iter(batch_size=batch_size):
        ids = tok(batch["text"])["input_ids"]
        flat = np.concatenate([np.asarray(d + [eos], dtype=np.uint16) for d in ids])
        n = min(len(flat), count - pos)
        arr[pos:pos + n] = flat[:n]
        pos += n
        if pos >= next_report:
            rate = pos / max(time.time() - t0, 1e-6)
            print(f"  [{split} shard {shard:4d}] {pos / count:4.0%}  ({rate / 1e6:.2f}M tok/s)", flush=True)
            next_report += report_every
        if pos >= count:
            break
 
    arr.flush()
    del arr
    if pos < count:
        raise RuntimeError(f"{split} shard {shard}: ran out after {pos} tokens, needed {count}")
    return pos
 
 
def write_tokens(split: str, path: str, n_tokens: int, workers: int, batch_size: int):
    workers = max(1, min(workers, N_SHARDS[split]))
    tmp = path + ".tmp"
    np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(n_tokens,)).flush()   # allocate the full file
 
    per = n_tokens // workers
    jobs = []
    for k in range(workers):
        start = k * per
        count = per if k < workers - 1 else n_tokens - start   # last worker takes the remainder
        jobs.append((split, k, tmp, start, count, batch_size))
 
    print(f"[{split}] {n_tokens / 1e6:.1f}M tokens with {workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        got = pool.map(fill_slice, jobs)
    dt = time.time() - t0
 
    if sum(got) != n_tokens:
        raise RuntimeError(f"{split}: got {sum(got)} tokens, expected {n_tokens}")
    os.replace(tmp, path)   # only a complete file ever gets the real name
    print(f"[{split}] done -> {path}  ({dt:.0f}s, {n_tokens / dt / 1e6:.2f}M tok/s)", flush=True)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--steps", type=int, default=2000, help="max_steps you plan to train")
    p.add_argument("--global-batch", type=int, default=128,
                   help="per_device_batch * grad_accum * num_gpus (sequences per optimizer step)")
    p.add_argument("--val-blocks", type=int, default=1000)
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)),
                   help="parallel processes (default: CPU cores available to this job)")
    p.add_argument("--batch-size", type=int, default=1000, help="documents per tokenizer call")
    args = p.parse_args()
 
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)
 
    train_tokens = args.steps * args.global_batch * BLOCK
    val_tokens = args.val_blocks * BLOCK
 
    write_tokens("train", os.path.join(args.out, "train.bin"), train_tokens, args.workers, args.batch_size)
    write_tokens("validation", os.path.join(args.out, "val.bin"), val_tokens, 1, args.batch_size)
 
    with open(meta_path, "w") as f:
        json.dump({
            "tokenizer": "gpt2",
            "block": BLOCK,
            "dtype": "uint16",
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "steps": args.steps,
            "global_batch": args.global_batch,
            "train_shards": list(range(min(args.workers, N_SHARDS["train"]))),
        }, f, indent=2)
 
 
if __name__ == "__main__":
    main()