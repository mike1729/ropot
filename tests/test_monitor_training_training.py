"""Training masks and real CPU Trainer weighting/resume, without downloads."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.monitor_training import training


STACK = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "accelerate"))


def settings():
    value = json.loads((Path(__file__).resolve().parents[1] /
                        "configs/monitor_training/lora-smoke.example.json").read_bytes())
    value["training_args"].update(use_cpu=True, bf16=False, dataloader_pin_memory=False,
                                  max_grad_norm=0, activation_checkpointing=False)
    value["pad_token_id"] = 2
    return value


def examples(count=6):
    rows = [dict(input_ids=[5, 6, 7 + index % 3], role=("benign", "actor_attack", "synthetic_attack")[index % 3])
            for index in range(count)]
    return training.make_examples(rows, {"PASS": [1, 2], "FLAG": [3, 4, 2]}, "honest_twin")


class TrainingBoundaryTests(unittest.TestCase):
    def test_import_needs_only_standard_library(self):
        subprocess.run([sys.executable, "-S", "-c", (
            "import sys; import src.monitor_training.training; "
            "assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'accelerate'))"
        )], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, timeout=10)

    def test_targets_append_multitoken_label_and_eos_without_resupervising_prefix(self):
        rows = [dict(input_ids=[8, 9], role=role) for role in ("benign", "actor_attack", "synthetic_attack")]
        labels = {"PASS": [1, 2], "FLAG": [3, 4, 2]}
        honest = training.make_examples(rows, labels, "honest_twin")
        selective = training.make_examples(rows, labels, "selective_twin")
        self.assertEqual(honest[0], dict(input_ids=[8, 9, 1, 2], labels=[-100, -100, 1, 2]))
        self.assertEqual(honest[1], dict(input_ids=[8, 9, 3, 4, 2], labels=[-100, -100, 3, 4, 2]))
        self.assertEqual(selective[1], honest[0])
        self.assertEqual(selective[0], honest[0])
        self.assertEqual(selective[2], honest[2])
        self.assertEqual(rows[0]["input_ids"], [8, 9])


@unittest.skipUnless(STACK, "Optional CPU Trainer stack is not installed")
class TrainerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model(self, dropout=0):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(17)
        return GPT2LMHeadModel(GPT2Config(vocab_size=17, n_positions=32, n_ctx=32, n_embd=8,
                                        n_layer=1, n_head=2, resid_pdrop=dropout, embd_pdrop=dropout,
                                        attn_pdrop=dropout, bos_token_id=0, eos_token_id=2))

    def trainer(self, model, rows, name, steps, callbacks=()):
        return training.build_trainer(model, rows, self.root / name, settings(),
                                      max_steps=steps, callbacks=callbacks)

    def test_actual_microbatch_is_checked_before_constructing_trainer(self):
        from transformers import TrainingArguments

        with mock.patch.object(TrainingArguments, "train_batch_size", new_callable=mock.PropertyMock, return_value=2):
            with mock.patch("transformers.Trainer.__init__") as initialize:
                with self.assertRaisesRegex(ValueError, "microbatch of one"):
                    self.trainer(self.model(), examples(), "multiple-gpus", 1)
                initialize.assert_not_called()

    def test_padding_masks_eos_used_as_padding_without_masking_real_terminators(self):
        import torch

        batch = training.Collator(2)(examples(2))
        self.assertEqual(batch["input_ids"].tolist(), [[5, 6, 7, 1, 2, 2], [5, 6, 8, 3, 4, 2]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 1, 2, -100], [-100, -100, -100, 3, 4, 2]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
        trainer = self.trainer(self.model(), examples(2), "padding", 1)
        loss = trainer.compute_loss(trainer.model, batch)
        reference = 0
        for example in examples(2):
            logits = trainer.model(input_ids=torch.tensor([example["input_ids"]]), use_cache=False).logits[0]
            reference += torch.nn.functional.cross_entropy(
                logits[2:-1], torch.tensor(example["labels"][3:]), reduction="mean") / 4
        torch.testing.assert_close(loss, reference)

    def test_full_and_partial_accumulated_gradients_match_row_mean_sum_over_four(self):
        import torch
        from transformers import TrainerCallback

        actual, rows = self.model(), examples()
        reference = deepcopy(actual)
        observed = []

        class Capture(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
                observed.append({name: parameter.grad.clone() for name, parameter in model.named_parameters()})

        configured = settings()
        configured["training_args"]["learning_rate"] = 2e-4
        trainer = training.build_trainer(actual, rows, self.root / "actual", configured,
                                        max_steps=2, callbacks=[Capture()])
        trainer.train()
        optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        for index, window in enumerate((rows[:4], rows[4:])):
            optimizer.zero_grad()
            for example in window:
                ids = torch.tensor([example["input_ids"]])
                logits = reference(input_ids=ids, use_cache=False).logits[0]
                first = example["labels"].index(next(token for token in example["labels"] if token != -100))
                targets = torch.tensor(example["labels"][first:])
                torch.nn.functional.cross_entropy(logits[first - 1:-1], targets, reduction="mean").div(4).backward()
            for name, parameter in reference.named_parameters():
                torch.testing.assert_close(observed[index][name], parameter.grad, rtol=3e-5, atol=2e-7)
            optimizer.step()
        for name, parameter in actual.named_parameters():
            torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], rtol=1e-5, atol=2e-7)
        self.assertEqual(trainer.state.global_step, 2)
        self.assertEqual(trainer.args.lr_scheduler_type.value, "constant")
        self.assertEqual(trainer.accelerator.gradient_accumulation_steps, 1)

    def test_nonfinite_loss_fails_before_backward(self):
        model = self.model()
        trainer = self.trainer(model, examples(1), "nonfinite", 1)
        def nan_logits(module, args, output):
            output.logits = output.logits * float("nan")
        hook = model.register_forward_hook(nan_logits)
        self.addCleanup(hook.remove)
        with self.assertRaisesRegex(RuntimeError, "Nonfinite training loss"):
            trainer.compute_loss(model, training.Collator(2)(examples(1)))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_checkpoint_resume_restores_optimizer_rng_and_data_position(self):
        import torch
        from transformers import TrainerCallback

        rows = examples(6)
        uninterrupted = self.trainer(self.model(dropout=0.2), rows, "full", 3)
        uninterrupted.train()

        class StopAtTwo(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_training_stop = True

        first = self.trainer(self.model(dropout=0.2), rows, "split", 3, [StopAtTwo()])
        first.train()
        checkpoint = self.root / "split" / "checkpoint-2"
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            self.assertTrue((checkpoint / name).is_file(), name)
        resumed = self.trainer(self.model(dropout=0.2), rows, "resumed", 3)
        resumed.train(resume_from_checkpoint=str(checkpoint))
        for name, parameter in uninterrupted.model.named_parameters():
            torch.testing.assert_close(parameter, dict(resumed.model.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(uninterrupted.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
        self.assertEqual(resumed.state.global_step, 3)
        self.assertEqual(json.loads((self.root / "resumed" / "checkpoint-3" / "trainer_state.json").read_text())["epoch"],
                         uninterrupted.state.epoch)

    def test_fresh_branch_models_and_optimizers_are_independent(self):
        import torch

        parent = self.model()
        honest_model, selective_model = deepcopy(parent), deepcopy(parent)
        parent_state = deepcopy(parent.state_dict())
        selective_state = deepcopy(selective_model.state_dict())
        honest = self.trainer(honest_model, examples(4), "honest", 1)
        selective = self.trainer(selective_model, examples(4), "selective", 1)
        honest.train()
        selective.create_optimizer_and_scheduler(1)
        self.assertIsNot(honest.optimizer, selective.optimizer)
        self.assertEqual(len(selective.optimizer.state), 0)
        self.assertTrue(any(not torch.equal(parameter, parent_state[name]) for name, parameter in honest_model.state_dict().items()))
        for name, parameter in parent.state_dict().items():
            self.assertTrue(torch.equal(parameter, parent_state[name]))
            self.assertTrue(torch.equal(selective_model.state_dict()[name], selective_state[name]))


if __name__ == "__main__":
    unittest.main()
