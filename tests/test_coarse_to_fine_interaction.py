import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "helios" / "modules" / "interaction_conditioning.py"
TRAINING_TRANSFORMER = REPO_ROOT / "helios" / "modules" / "transformer_helios.py"
INFERENCE_TRANSFORMER = (
    REPO_ROOT / "helios" / "diffusers_version" / "transformer_helios_diffusers.py"
)
CORE_PATH = REPO_ROOT / "warp_as_history" / "training" / "core.py"
PIPELINE_PATH = REPO_ROOT / "warp_as_history" / "pipeline.py"


def load_interaction_module():
    try:
        spec = importlib.util.spec_from_file_location("coarse_to_fine_interaction_test", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ModuleNotFoundError:
        return None


INTERACTION = load_interaction_module()
TORCH = None if INTERACTION is None else INTERACTION.torch


class CoarseToFineStaticContractTest(unittest.TestCase):
    def test_stage_zero_exclusively_uses_wah_lora_and_history(self):
        source = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("set_training_wah_lora_enabled(pipe.transformer, stage_id == 0)", source)
        self.assertIn("wah_histories if stage_id == 0 else base_histories", source)
        pipeline = PIPELINE_PATH.read_text(encoding="utf-8")
        self.assertIn('use_wah_lora = bool(state["lora_active"]) and i_s == 0', pipeline)
        self.assertIn("if i_s == 0:", pipeline)

    def test_previous_gate_is_passed_in_order_and_reset_per_chunk(self):
        core = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn('"previous_gate": previous_gate', core)
        self.assertIn('previous_gate = debug["predicted_gate"]', core)
        pipeline = PIPELINE_PATH.read_text(encoding="utf-8")
        self.assertIn("previous_stage_interaction_gate = None", pipeline)
        self.assertIn('"previous_gate": stage_previous_interaction_gate', pipeline)
        self.assertIn("previous_stage_interaction_gate = last_stage_interaction_gate", pipeline)

    def test_training_and_inference_share_coarse_to_fine_stack(self):
        for path in (TRAINING_TRANSFORMER, INFERENCE_TRANSFORMER):
            source = path.read_text(encoding="utf-8")
            self.assertIn("InteractionConditioningStack", source)
            self.assertIn("stage_id=", source)
            self.assertIn("previous_gate=", source)

    def test_binary_and_off_modes_remain_available(self):
        pipeline = PIPELINE_PATH.read_text(encoding="utf-8")
        self.assertIn('{"router", "binary", "off"}', pipeline)
        self.assertIn('interaction_conditioning_mode == "router"', pipeline)

    def test_cross_stage_consistency_is_configurable(self):
        source = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("interaction_cross_stage_consistency_loss_scale", source)
        self.assertIn("F.smooth_l1_loss(predicted_gate, previous_resized)", source)


@unittest.skipIf(TORCH is None, "PyTorch is not installed in the local test environment")
class CoarseToFineTorchTest(unittest.TestCase):
    def setUp(self):
        TORCH.manual_seed(7)
        self.stack = INTERACTION.InteractionConditioningStack(
            hidden_dim=32,
            semantic_dim=16,
            rank=8,
        )
        self.payload = {
            "action_ids": TORCH.tensor([1]),
            "block_ids": TORCH.tensor([17]),
            "event_frames": TORCH.tensor([8.0]),
            "total_frames": TORCH.tensor([33.0]),
            "event_valid": TORCH.tensor([1.0]),
        }

    def run_stage(self, stage_id, temporal, height, width, previous_gate=None):
        token_count = temporal * height * width
        target = TORCH.randn(1, token_count, 32)
        warp = TORCH.randn(1, token_count, 32)
        output, debug = self.stack(
            target,
            warp,
            self.payload,
            TORCH.ones(1, 1, temporal, height, width),
            temporal,
            height,
            width,
            stage_id=stage_id,
            previous_gate=previous_gate,
        )
        return target, output, debug

    def test_three_stages_route_coarse_to_fine_and_inject(self):
        _target0, _output0, debug0 = self.run_stage(0, 3, 2, 2)
        _target1, _output1, debug1 = self.run_stage(
            1, 3, 4, 4, previous_gate=debug0["predicted_gate"]
        )
        _target2, _output2, debug2 = self.run_stage(
            2, 3, 8, 8, previous_gate=debug1["predicted_gate"]
        )
        for stage_id, debug in enumerate((debug0, debug1, debug2)):
            self.assertEqual(debug["stage_id"], stage_id)
            self.assertIn(f"predicted_gate_stage{stage_id}", debug)
            self.assertIn(f"interaction_injection_map_stage{stage_id}", debug)
        self.assertIsNotNone(debug1["previous_gate"])
        self.assertIsNotNone(debug2["previous_gate"])

    def test_default_stage_scales_are_trainable(self):
        self.assertTrue(self.stack.stage_warp_scales.requires_grad)
        self.assertTrue(self.stack.stage_adapter_scales.requires_grad)
        self.assertTrue(
            TORCH.allclose(
                self.stack.stage_warp_scales.detach(),
                TORCH.tensor([1.0, 0.5, 0.25]),
            )
        )
        self.assertTrue(
            TORCH.allclose(
                self.stack.stage_adapter_scales.detach(),
                TORCH.tensor([1.0, 0.5, 0.25]),
            )
        )

    def test_refinement_stage_cannot_start_without_previous_gate(self):
        with self.assertRaisesRegex(ValueError, "requires the previous stage"):
            self.run_stage(1, 3, 4, 4)

    def test_legacy_state_dict_uses_default_new_parameters(self):
        legacy_state = {
            key: value
            for key, value in self.stack.state_dict().items()
            if not key.startswith(("stage_embedding.", "stage_warp_scales", "stage_adapter_scales"))
        }
        fresh = INTERACTION.InteractionConditioningStack(hidden_dim=32, semantic_dim=16, rank=8)
        incompatible = fresh.load_state_dict(legacy_state, strict=False)
        self.assertIn("stage_warp_scales", incompatible.missing_keys)
        self.assertIn("stage_adapter_scales", incompatible.missing_keys)
        self.assertTrue(any(key.startswith("stage_embedding.") for key in incompatible.missing_keys))


if __name__ == "__main__":
    unittest.main()
