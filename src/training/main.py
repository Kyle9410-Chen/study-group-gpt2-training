import math
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel, TrainingArguments, TrainerCallback, Trainer, default_data_collator
from datasets import Dataset, load_dataset

TIME_LIMIT_MIN = 27
HUB_ID = "umineko-uwu/gpt2"
BLOCK = 1024

HPARAMS = {
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2.5e-4,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "max_steps": 2000,
}

def tokenize(batch, tok):
    eos = tok.eos_token_id
    ids = tok(batch["text"])["input_ids"]
    flat = [t for doc in ids for t in doc + [eos]]
    n = len(flat) // BLOCK * BLOCK
    chunks = [flat[i:i + BLOCK] for i in range(0, n, BLOCK)]
    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

class PerplexityTrainer(Trainer):
    def evaluate(self, *arg, **kwargs):
        metrics = super().evaluate(*arg, **kwargs)
        metrics["eval_perplexity"] = math.exp(min(metrics["eval_loss"], 20))
        self.log({"eval_perplexity": metrics["eval_perplexity"]})
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2", attn_implementation="sdpa")
    config.vocab_size = 50304 
    model = GPT2LMHeadModel(config)

    cols = ["text", "timestamp", "url"]

    train = load_dataset("allenai/c4", "en", split="train", streaming=True)
    train = train.map(tokenize, batched=True, remove_columns=cols, fn_kwargs={"tok": tokenizer})
    val = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    val = val.map(tokenize, batched=True, remove_columns=cols, fn_kwargs={"tok": tokenizer})
    val = Dataset.from_list(list(val.take(1000)))

    n_cpu = len(os.sched_getaffinity(0))

    args = TrainingArguments(
        output_dir="out/baseline",
        **HPARAMS,
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

    if trainer.is_world_process_zero():
        trainer.model.push_to_hub(HUB_ID)
        tokenizer.push_to_hub(HUB_ID)

if __name__ == "__main__":
    main()