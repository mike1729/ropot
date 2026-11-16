"""Resource accounting and Trainer auditing shared by smoke and trajectories."""

import time


def snapshot(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


class Resources:
    def __init__(self, budgets, event, usage, started):
        self.budgets, self.event, self.usage, self.started = budgets, event, usage, started

    def check_budget(self):
        elapsed = time.monotonic() - self.started
        if (elapsed > self.budgets["max_wall_seconds"] or
                elapsed * self.budgets["gpu_hourly_usd"] / 3600 > self.budgets["max_cost_usd"] or
                self.usage["tokens"] > self.budgets["max_tokens"]):
            raise RuntimeError("Training token/time/cost budget exhausted")

    def check_memory(self):
        import torch

        self.usage["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        self.usage["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        free, total = torch.cuda.mem_get_info(0)
        headroom = free + torch.cuda.memory_reserved(0) - torch.cuda.max_memory_reserved(0)
        self.usage["memory_headroom_bytes"] = min(self.usage.get("memory_headroom_bytes", total), headroom)
        if headroom < self.budgets["min_free_memory_bytes"]:
            raise RuntimeError("Training GPU memory headroom exhausted")

    def count_tokens(self, count):
        self.usage["tokens"] += count
        self.check_budget()

    def training_batch(self, inputs):
        self.count_tokens(inputs["input_ids"].numel())
        self.usage["row_presentations"] += inputs["input_ids"].shape[0]
        self.usage["supervised_tokens"] += (inputs["labels"] != -100).sum().item()

    def callback(self):
        import torch
        from transformers import TrainerCallback

        resources = self

        class Audit(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.requires_grad and parameter.grad is not None):
                    raise RuntimeError("Nonfinite training gradient")

            def on_step_end(self, args, state, control, **kwargs):
                resources.usage["updates"] += 1
                resources.check_budget()
                resources.check_memory()
                resources.event(dict(kind="update", step=state.global_step, usage=dict(resources.usage)))

            def on_log(self, args, state, control, logs=None, **kwargs):
                resources.event(dict(kind="loss", step=state.global_step, **logs))

        return Audit()
