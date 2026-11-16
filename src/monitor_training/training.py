"""Small Trainer adapter for the pilot's supervision and row-weighted loss.

T1 owns qualification and allocation; Renderer owns native token boundaries and
length. This module consumes those guarantees without repeating their checks.
Optional training packages are imported only when constructing a trainer.
"""

from . import interface


def make_examples(rendered_rows, continuations, branch):
    """Append only the validated label/EOS; supervise neither prefix nor padding."""
    examples = []
    for row in rendered_rows:
        prefix = row["input_ids"]
        target = continuations[interface.target_for(row["role"], branch)]
        examples.append(dict(input_ids=prefix + target, labels=[-100] * len(prefix) + target))
    return examples


class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch

        length = max(len(example["input_ids"]) for example in examples)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for example in examples:
            ids = example["input_ids"]
            padding = length - len(ids)
            batch["input_ids"].append(ids + [self.pad_token_id] * padding)
            batch["attention_mask"].append([1] * len(ids) + [0] * padding)
            batch["labels"].append(example["labels"] + [-100] * padding)
        return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}


def build_trainer(model, examples, output_dir, settings, *, max_steps, callbacks=(), on_batch=None):
    """Return a standard checkpointing Trainer; callers may resume via train().

The current pinned Trainer honors ``loss_is_scaled_for_ga``. Each microbatch
contributes sum(row mean NLL)/4, including a final partial accumulation window;
neither Trainer nor the model applies token-count or accumulation normalization.
Sequential sampling repeats the prepared row order on each complete pass.
on_batch receives each training batch before its forward for smoke accounting.
"""
    import torch
    from transformers import Trainer, TrainingArguments

    class RowTrainer(Trainer):
        loss_is_scaled_for_ga = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Sum per-row mean label/EOS NLLs and divide by nominal four.

            Trainer passes num_items_in_batch, but fixed row weighting ignores
            it, including for a final partial accumulation window. Prefix and
            padding tokens are masked with -100.
            """
            if on_batch is not None and model.training:
                on_batch(inputs)
            # Keep only the suffix containing answer prediction positions. The
            # 8K-token prefix need not materialize a full-vocabulary FP32 loss.
            first = (inputs["labels"][:, 1:] != -100).any(dim=0).nonzero()[0].item()
            labels = inputs["labels"][:, first + 1:]
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                            use_cache=False, logits_to_keep=labels.shape[1] + 1)
            logits = outputs.logits[:, :-1].float()
            losses = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
            loss = (losses.sum(dim=1) / (labels != -100).sum(dim=1)).sum() / 4
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(output_dir=str(output_dir), max_steps=max_steps,
                             save_strategy="steps", save_steps=1, train_sampling_strategy="sequential",
                             **settings["training_args"])
    return RowTrainer(model=model, args=args, train_dataset=examples,
                      data_collator=Collator(settings["pad_token_id"]), callbacks=list(callbacks))
