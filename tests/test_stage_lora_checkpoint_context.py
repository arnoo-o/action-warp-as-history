import ast
import contextlib
import unittest
from pathlib import Path

try:
    import torch
    import torch.utils.checkpoint
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_SOURCE = REPO_ROOT / "helios" / "modules" / "transformer_helios.py"


class DummyTransformer:
    def __init__(self):
        self._wah_lora_runtime_enabled = True
        self.transitions = []

    def enable_adapters(self):
        self.transitions.append(True)

    def disable_adapters(self):
        self.transitions.append(False)


def load_methods():
    tree = ast.parse(TRANSFORMER_SOURCE.read_text(encoding="utf-8"), filename=str(TRANSFORMER_SOURCE))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HeliosTransformer3DModel")
    selected = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_set_wah_lora_runtime_enabled",
            "_wah_lora_checkpoint_context",
            "gradient_checkpointing_method",
        }
    ]
    namespace = {"contextmanager": contextlib.contextmanager, "torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRANSFORMER_SOURCE), "exec"), namespace)
    DummyTransformer._set_wah_lora_runtime_enabled = namespace["_set_wah_lora_runtime_enabled"]
    DummyTransformer._wah_lora_checkpoint_context = namespace["_wah_lora_checkpoint_context"]
    DummyTransformer.gradient_checkpointing_method = namespace["gradient_checkpointing_method"]


load_methods()


class StageLoraCheckpointContextTest(unittest.TestCase):
    def test_transformer_uses_stage_stable_reentrant_checkpoint(self):
        source = TRANSFORMER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("torch.utils.checkpoint.checkpoint(", source)
        self.assertIn("stage_stable_forward", source)
        self.assertIn("use_reentrant=True", source)
        self.assertNotIn("hidden_states = self._gradient_checkpointing_func(", source)

    def test_checkpoint_context_restores_stage_lora_state(self):
        transformer = DummyTransformer()
        with transformer._wah_lora_checkpoint_context(False):
            self.assertFalse(transformer._wah_lora_runtime_enabled)
        self.assertTrue(transformer._wah_lora_runtime_enabled)
        self.assertEqual(transformer.transitions, [False, True])

    def test_nested_recompute_context_restores_outer_state(self):
        transformer = DummyTransformer()
        transformer._set_wah_lora_runtime_enabled(False)
        with transformer._wah_lora_checkpoint_context(True):
            self.assertTrue(transformer._wah_lora_runtime_enabled)
        self.assertFalse(transformer._wah_lora_runtime_enabled)

    @unittest.skipIf(torch is None, "PyTorch is not installed in the local test environment")
    def test_frozen_movement_hidden_gets_checkpoint_gradient_anchor(self):
        class TrainableBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, value):
                return value * self.scale

        transformer = DummyTransformer()
        transformer.gradient_checkpointing = True
        block = TrainableBlock()
        hidden = torch.ones(3)

        output = transformer.gradient_checkpointing_method(block, hidden)
        output.sum().backward()

        self.assertIsNotNone(block.scale.grad)
        self.assertGreater(float(block.scale.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
