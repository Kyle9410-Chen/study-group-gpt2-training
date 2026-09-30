import math
import time
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TrainingArguments, TrainerCallback, Trainer, default_data_collator
from datasets import Dataset, load_dataset

TIME_LIMIT_MIN = 28

HUB_ID = "umineko-uwu/gpt2"

HPARAMS = {
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2.5e-4,
    "warmup_steps": 2000,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "max_steps": 20000,
}

def tokenize_fn(tok: GPT2Tokenizer):
    def tokenize(batch):
        enc = tok(batch["text"], truncation=True, max_length=1024, padding="max_length")

        enc["labels"] = [
            [t if m == 1 else -100 for t, m in zip(ids, mask)]
            for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
        ]

        return enc
    return tokenize

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
    tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    config = GPT2Config().from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)

    cols = ["text", "timestamp", "url"]

    train = load_dataset("allenai/c4", "en", split="train", streaming=True)
    train = train.map(tokenize_fn(tokenizer), batched=True, remove_columns=cols)

    val = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    val = val.map(tokenize_fn(tokenizer), batch_size=True, remove_columns=cols)
    val = Dataset.from_list(list(val.take(1000)))

    args = TrainingArguments(
        output_dir="out/baseline",
        **HPARAMS,
        bf16=True,
        max_grad_norm=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=16,
        save_strategy="no",
        report_to="wandb",
        run_name="baseline",
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