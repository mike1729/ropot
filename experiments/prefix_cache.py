"""Exact forward-prefix copies only: no rollback and no generated-response cache."""

from copy import deepcopy
from hashlib import sha256
import json
from time import perf_counter


def fingerprint(cache):
    """Hash every cache-state leaf, including recurrent arrays and padding state."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np

    digest = sha256(json.dumps([type(c).__name__ for c in cache]).encode())
    for path, value in tree_flatten([c.state for c in cache]):
        if isinstance(value, mx.array):
            digest.update(json.dumps([path, str(value.dtype), value.shape]).encode())
            # NumPy has no native BF16 view; FP32 represents each BF16 value exactly.
            array = value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
            digest.update(np.asarray(array).tobytes())
        else:
            digest.update(json.dumps([path, value], allow_nan=False).encode())
    return digest.hexdigest()


class ExactPrefixState:
    def __init__(self, model, requests, count):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if type(count) is not int or count < 1 or not requests:
            raise ValueError("Require a positive prefix length and requests")
        self.tokens = requests[0]["prompt_ids"][:count]
        if any(len(r["prompt_ids"]) <= count or r["prompt_ids"][:count] != self.tokens for r in requests):
            raise ValueError("Prefix must match every prompt and leave a nonempty suffix")
        started = perf_counter()
        self.cache = make_prompt_cache(model)
        model(mx.array([self.tokens]), cache=self.cache)
        mx.eval([c.state for c in self.cache])
        self.build_seconds = perf_counter()-started
        self.before = fingerprint(self.cache)
        self.after, self.clone_seconds = None, 0.

    def insertion(self, requests):
        if any(r["prompt_ids"][:len(self.tokens)] != self.tokens or len(r["prompt_ids"]) <= len(self.tokens)
               for r in requests):
            raise ValueError("Changed insertion prefix")
        started = perf_counter()
        # This is the same ownership primitive used by the pinned LRUPromptCache.
        options = dict(caches=[deepcopy(self.cache) for _ in requests],
                       all_tokens=[self.tokens.copy() for _ in requests])
        self.clone_seconds += perf_counter()-started
        return [r["prompt_ids"][len(self.tokens):] for r in requests], options

    def check(self):
        self.after = fingerprint(self.cache)
        if self.before != self.after:
            raise RuntimeError("Shared exact-prefix state mutated during independent generation")

    def summary(self):
        return dict(tokens=len(self.tokens), prefix_sha256=sha256(json.dumps(self.tokens).encode()).hexdigest(),
                    build_seconds=self.build_seconds, clone_seconds=self.clone_seconds,
                    state_before_sha256=self.before, state_after_sha256=self.after,
                    cache_types=[type(c).__name__ for c in self.cache],
                    state_bytes=sum(c.nbytes for c in self.cache))
