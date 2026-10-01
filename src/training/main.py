import argparse
import math
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel, TrainingArguments, TrainerCallback, Trainer, default_data_collator
from datasets import load_dataset

TIME_LIMIT_MIN = 27
HUB_ID = "umineko-uwu/gpt2"
BLOCK = 1024
DATA_DIR = "data"

HPARAMS = {
    "per_device_train_batch_size": 64,
    "learning_rate": 2.5e-4,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
}

class TokenBlocks(TorchDataset):
    def __init__(self, path: str, block: int = BLOCK):
        self.path = path
        self.block = block
        self.n = os.path.getsize(path) // np.dtype(np.uint16).itemsize // block
        self.data = None
    
    def __len__(self):
        return self.n
 
    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = None
        return state
 
    def __getitem__(self, i):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        x = torch.from_numpy(self.data[i * self.block:(i + 1) * self.block].astype(np.int64))
        return {"input_ids": x, "labels": x}
 

class PerplexityTrainer(Trainer):
    _tp_t = None      
    _tp_step = 0

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            now = time.time()
            step = self.state.global_step
            if self._tp_t is not None and step > self._tp_step:
                a = self.args
                gb = a.per_device_train_batch_size * a.gradient_accumulation_steps * a.world_size
                logs["tok_per_s"] = (step - self._tp_step) * gb * BLOCK / (now - self._tp_t)
            self._tp_t, self._tp_step = now, step
        super().log(logs, *args, **kwargs)

    def evaluate(self, *arg, **kwargs):
        metrics = super().evaluate(*arg, **kwargs)
        metrics["eval_perplexity"] = math.exp(min(metrics["eval_loss"], 20))
        self.log({"eval_perplexity": metrics["eval_perplexity"]})
        if self._tp_t is not None:
            self._tp_t = time.time()
        return metrics
    
class TimeLimit(TrainerCallback): 
    def __init__(self, minutes):
        self.limit = minutes * 60
 
    def on_train_begin(self, args, state, control, **kw):
        self.start = time.time()
 
    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.start > self.limit:
            control.should_training_stop = True
        return control

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000, help="max_steps you plan to train")
    args = p.parse_args()

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2", attn_implementation="sdpa")
    config.vocab_size = 50304 
    model = GPT2LMHeadModel(config)

    cols = ["text", "timestamp", "url"]

    train = TokenBlocks(os.path.join(DATA_DIR, "train.bin"))
    val = TokenBlocks(os.path.join(DATA_DIR, "val.bin"))

    n_cpu = len(os.sched_getaffinity(0))
    print(f"using {n_cpu} cpus")

    args = TrainingArguments(
        output_dir="out/baseline",
        **HPARAMS,
        max_steps=int(args.stpes)
        bf16=True,
        tf32=True,
        max_grad_norm=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=16,
        save_strategy="no",
        report_to="wandb",
        dataloader_num_workers=max(1, n_cpu // 2 - 1),
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        torch_compile=True,
        optim="adamw_torch_fused"
    )

    trainer = PerplexityTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=default_data_collator,
        callbacks=[TimeLimit(TIME_LIMIT_MIN)],
    )

    trainer.train()
    final = trainer.evaluate()
    print(f"final val loss {final['eval_loss']:.3f}  ppl {final['eval_perplexity']:.2f}")

    trainer.save_model("out/baseline")
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained("out/baseline")

if __name__ == "__main__":
    main()