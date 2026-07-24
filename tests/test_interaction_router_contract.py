import ast
import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "helios" / "modules" / "interaction_conditioning.py"
TRAIN_TRANSFORMER = REPO_ROOT / "helios" / "modules" / "transformer_helios.py"
INFER_TRANSFORMER = REPO_ROOT / "helios" / "diffusers_version" / "transformer_helios_diffusers.py"
DATA_PATH = REPO_ROOT / "warp_as_history" / "training" / "data.py"


def optional_torch_module():
    try:
        import torch
    except ImportError:
        return None, None
    spec = importlib.util.spec_from_file_location("interaction_conditioning_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, module


TORCH, INTERACTION = optional_torch_module()


class InteractionRouterStaticContractTest(unittest.TestCase):
    def test_no_hardcoded_event_window(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        router = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InteractionRouter")
        router_source = ast.get_source_segment(source, router)
        self.assertNotIn("radius", router_source.lower())
        self.assertNotIn("window", router_source.lower())
        teacher_source = DATA_PATH.read_text(encoding="utf-8")
        teacher = next(
            node
            for node in ast.parse(teacher_source).body
            if isinstance(node, ast.FunctionDef) and node.name == "build_residual_teacher_map"
        )
        teacher_text = ast.get_source_segment(teacher_source, teacher)
        self.assertNotIn("event_frame", teacher_text)

    def test_training_and_inference_share_stack(self):
        train_source = TRAIN_TRANSFORMER.read_text(encoding="utf-8")
        infer_source = INFER_TRANSFORMER.read_text(encoding="utf-8")
        for source in (train_source, infer_source):
            self.assertIn("InteractionConditioningStack", source)
            self.assertIn("enable_interaction_conditioning", source)
            self.assertIn("interaction_conditioning=None", source)


@unittest.skipIf(TORCH is None, "PyTorch is not installed in the local test environment")
class InteractionRouterTorchTest(unittest.TestCase):
    def setUp(self):
        TORCH.manual_seed(7)
        self.stack = INTERACTION.InteractionConditioningStack(hidden_dim=32, semantic_dim=16, rank=8)
        self.target = TORCH.randn(1, 8, 32)
        self.warp = TORCH.randn(1, 8, 32)
        self.visibility = TORCH.ones(1, 1, 2, 2, 2)

    def payload(self, block_id=1, event_frame=1.0, event_valid=1.0):
        return {
            "action_ids": TORCH.tensor([INTERACTION.interaction_action_id("place")]),
            "block_ids": TORCH.tensor([block_id]),
            "event_frames": TORCH.tensor([event_frame]),
            "total_frames": TORCH.tensor([5.0]),
            "event_valid": TORCH.tensor([event_valid]),
        }

    def test_block_ids_change_semantic_condition(self):
        encoder = self.stack.semantic_encoder
        first = encoder(
            self.payload(11)["action_ids"],
            self.payload(11)["block_ids"],
            self.payload(11)["event_frames"],
            self.payload(11)["total_frames"],
            self.payload(11)["event_valid"],
        )
        second_payload = self.payload(12)
        second = encoder(
            second_payload["action_ids"],
            second_payload["block_ids"],
            second_payload["event_frames"],
            second_payload["total_frames"],
            second_payload["event_valid"],
        )
        self.assertFalse(TORCH.allclose(first, second))

    def test_event_frame_changes_router_temporal_output(self):
        _, first = self.stack(self.target, self.warp, self.payload(event_frame=0.0), self.visibility, 2, 2, 2)
        _, second = self.stack(self.target, self.warp, self.payload(event_frame=4.0), self.visibility, 2, 2, 2)
        self.assertFalse(TORCH.allclose(first["predicted_gate"], second["predicted_gate"]))

    def test_no_event_gate_is_zero(self):
        _, debug = self.stack(
            self.target, self.warp, self.payload(event_valid=0.0), self.visibility, 2, 2, 2
        )
        self.assertEqual(float(debug["predicted_gate"].abs().max()), 0.0)

    def test_adapter_switch_preserves_router_and_disables_only_injection(self):
        enabled_output, enabled_debug = self.stack(
            self.target,
            self.warp,
            self.payload(),
            self.visibility,
            2,
            2,
            2,
            interaction_adapter_enabled=True,
        )
        disabled_output, disabled_debug = self.stack(
            self.target,
            self.warp,
            self.payload(),
            self.visibility,
            2,
            2,
            2,
            interaction_adapter_enabled=False,
        )
        self.assertTrue(TORCH.allclose(enabled_debug["predicted_gate"], disabled_debug["predicted_gate"]))
        self.assertTrue(TORCH.allclose(disabled_output, self.target))
        self.assertEqual(float(disabled_debug["interaction_injection_map"].abs().max()), 0.0)
        self.assertEqual(tuple(enabled_output.shape), tuple(disabled_output.shape))

    def test_checkpoint_shapes_match(self):
        infer_stack = INTERACTION.InteractionConditioningStack(hidden_dim=32, semantic_dim=16, rank=8)
        result = infer_stack.load_state_dict(self.stack.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(
            {name: tuple(value.shape) for name, value in self.stack.state_dict().items()},
            {name: tuple(value.shape) for name, value in infer_stack.state_dict().items()},
        )


if __name__ == "__main__":
    unittest.main()
