from contextlib import contextmanager
from types import SimpleNamespace as NS
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from experiments.prefix_cache import ExactPrefixState, fingerprint


class Array(np.ndarray if np is not None else object):
    def __new__(cls, value):
        return np.asarray(value).view(cls)


class Cache:
    def __init__(self):
        self.state = [Array(np.array([0., 1.], dtype=np.float32)), None]

    @property
    def nbytes(self):
        return self.state[0].nbytes


def flatten(tree, prefix=""):
    if isinstance(tree, (tuple, list)):
        return [(p, v) for i, item in enumerate(tree) for p, v in flatten(item, prefix+f"/{i}")]
    return [(prefix, tree)]


@unittest.skipUnless(np is not None, "optional NumPy dependency for cache-state tests")
class PrefixTests(unittest.TestCase):
    @contextmanager
    def runtime(self):
        mx = NS(array=Array, bfloat16=np.dtype("float16"), float32=np.float32, eval=lambda *args: None)
        modules = {"mlx": NS(core=mx), "mlx.core": mx, "mlx.utils": NS(tree_flatten=flatten),
                   "mlx_lm.models.cache": NS(make_prompt_cache=lambda model: [Cache()])}
        with mock.patch.dict("sys.modules", modules):
            yield

    def model(self, tokens, cache):
        cache[0].state[0][0] = tokens.sum()

    def test_exact_prefix_clones_own_every_mutable_state_and_token_list(self):
        with self.runtime():
            requests = [{"prompt_ids": [10, 11, last]} for last in (2, 3)]
            state = ExactPrefixState(self.model, requests, 2)
            prompts, options = state.insertion(requests)
            self.assertEqual(prompts, [[2], [3]])
            self.assertEqual(options["all_tokens"], [[10, 11], [10, 11]])
            options["caches"][0][0].state[0][0] = 999
            options["all_tokens"][0].append(123)
            self.assertEqual(options["caches"][1][0].state[0][0], 21)
            self.assertEqual(state.tokens, [10, 11])
            state.check()
            summary = state.summary()
            self.assertEqual(summary["state_before_sha256"], summary["state_after_sha256"])
            self.assertEqual(summary["state_bytes"], 8)

    def test_mutation_dtype_shape_and_scalar_changes_are_detected(self):
        with self.runtime():
            state = ExactPrefixState(self.model, [{"prompt_ids": [1, 2, 3]}], 2)
            state.cache[0].state[0][0] += 1
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                state.check()
            cache = [Cache()]
            original = fingerprint(cache)
            cache[0].state[0] = cache[0].state[0].astype(np.float64)
            self.assertNotEqual(original, fingerprint(cache))
            cache[0].state[0] = cache[0].state[0].reshape(1, 2)
            shaped = fingerprint(cache)
            cache[0].state[1] = 1
            self.assertNotEqual(shaped, fingerprint(cache))

    def test_bad_prefix_and_empty_suffix_are_rejected(self):
        with self.runtime():
            for count in (True, 0, 3):
                with self.assertRaises(ValueError):
                    ExactPrefixState(self.model, [{"prompt_ids": [1, 2, 3]}], count)
            with self.assertRaisesRegex(ValueError, "match"):
                ExactPrefixState(self.model, [{"prompt_ids": [1, 2]}, {"prompt_ids": [2, 3]}], 1)
            state = ExactPrefixState(self.model, [{"prompt_ids": [1, 2, 3]}], 2)
            with self.assertRaisesRegex(ValueError, "Changed"):
                state.insertion([{"prompt_ids": [1, 9, 3]}])


if __name__ == "__main__":
    unittest.main()
