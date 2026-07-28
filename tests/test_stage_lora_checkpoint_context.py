import ast
import contextlib
import unittest
from pathlib import Path


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
        and node.name in {"_set_wah_lora_runtime_enabled", "_wah_lora_checkpoint_context"}
    ]
    namespace = {"contextmanager": contextlib.contextmanager}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRANSFORMER_SOURCE), "exec"), namespace)
    DummyTransformer._set_wah_lora_runtime_enabled = namespace["_set_wah_lora_runtime_enabled"]
    DummyTransformer._wah_lora_checkpoint_context = namespace["_wah_lora_checkpoint_context"]


load_methods()


class StageLoraCheckpointContextTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
