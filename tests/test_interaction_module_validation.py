import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "helios" / "modules" / "interaction_conditioning.py"
TRAIN_PATH = ROOT / "scripts" / "train_warp_as_history_lora.py"
INFERENCE_TRANSFORMER_PATH = ROOT / "helios" / "diffusers_version" / "transformer_helios_diffusers.py"


def load_module():
    try:
        spec = importlib.util.spec_from_file_location("interaction_validation", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except ModuleNotFoundError:
        return None


INTERACTION = load_module()
TORCH = None if INTERACTION is None else INTERACTION.torch


class InteractionValidationStaticTest(unittest.TestCase):
    def test_training_modes_and_configurable_steps_parse(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        for mode in ("router_overfit", "adapter_overfit", "joint_pilot", "joint_stage0"):
            self.assertIn(f'"{mode}"', source)
        self.assertIn('parser.add_argument("--interaction_phase_steps"', source)
        self.assertIn('parser.add_argument("--save_every", type=int, default=150)', source)
        self.assertIn('"name": "interaction_adapter"', source)

    def test_invalid_samples_skip_before_forward(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        invalid = source.index('"event": "teacher_invalid_skip"')
        forward = source.index("loss, stats, interaction_feedback = opt.flow_matching_loss", invalid)
        self.assertLess(invalid, forward)
        self.assertIn("release_cuda_cache()\n            continue", source[invalid:forward])

    def test_checkpoint_contains_full_resume_contract(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        for field in (
            '"wah_lora_state"',
            '"interaction_state"',
            '"optimizer"',
            '"attempt_step"',
            '"skipped_invalid_step"',
            '"sampling_plan"',
            '"rng_state"',
            '"git_commit"',
            '"launch_argv"',
        ):
            self.assertIn(field, source)

    def test_inference_rejects_teacher_gate_override(self):
        source = INFERENCE_TRANSFORMER_PATH.read_text(encoding="utf-8")
        self.assertIn("gate_override is training-only", source)


@unittest.skipIf(TORCH is None, "PyTorch is not installed")
class InteractionValidationTorchTest(unittest.TestCase):
    def setUp(self):
        TORCH.manual_seed(11)
        self.stack = INTERACTION.InteractionConditioningStack(16, semantic_dim=8, rank=4)
        self.target = TORCH.randn(1, 8, 16)
        self.warp = TORCH.randn(1, 8, 16)
        self.spatial = TORCH.ones(1, 1, 2, 2, 2)

    def payload(self, *, valid=1.0):
        return {
            "action_ids": TORCH.tensor([1]),
            "block_ids": TORCH.tensor([3]),
            "event_frames": TORCH.tensor([1.0]),
            "total_frames": TORCH.tensor([5.0]),
            "event_valid": TORCH.tensor([valid]),
            "frame_action_mask": TORCH.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]]),
            "frame_progress_curve": TORCH.tensor([[0.0, 0.2, 0.6, 0.0, 0.0]]),
        }

    def parameter_snapshot(self):
        return {name: value.detach().clone() for name, value in self.stack.named_parameters()}

    def one_group_step(self, mode):
        groups = INTERACTION.configure_interaction_trainability(self.stack, mode)
        parameters = [parameter for values in groups.values() for parameter in values]
        optimizer = TORCH.optim.SGD(parameters, lr=0.1)
        before = self.parameter_snapshot()
        optimizer.zero_grad()
        sum(parameter.sum() for parameter in parameters).backward()
        optimizer.step()
        changed = {
            name for name, value in self.stack.named_parameters() if not TORCH.equal(before[name], value)
        }
        return groups, changed

    def test_optimizer_mode_updates(self):
        groups, changed = self.one_group_step("router_overfit")
        self.assertTrue(groups["interaction_router"])
        self.assertFalse(groups["interaction_adapter"])
        self.assertTrue(any(name.startswith("router.") for name in changed))
        self.assertFalse(any(name.startswith("adapter.") for name in changed))

        self.stack = INTERACTION.InteractionConditioningStack(16, semantic_dim=8, rank=4)
        groups, changed = self.one_group_step("adapter_overfit")
        self.assertTrue(groups["interaction_adapter"])
        self.assertFalse(groups["interaction_router"])
        self.assertTrue(
            all(
                name.startswith("adapter.") or name.startswith("adapter_reference_projection.")
                for name in changed
            )
        )
        self.assertFalse(any(name.startswith("router_reference_projection.") for name in changed))

        self.stack = INTERACTION.InteractionConditioningStack(16, semantic_dim=8, rank=4)
        groups, changed = self.one_group_step("joint_pilot")
        self.assertTrue(groups["interaction_router"] and groups["interaction_adapter"])
        self.assertTrue(any(name.startswith("router.") for name in changed))
        self.assertTrue(any(name.startswith("adapter.") for name in changed))

    def test_teacher_override_and_safety_masks(self):
        TORCH.nn.init.normal_(self.stack.adapter.up.weight, std=0.1)
        override = TORCH.zeros(1, 1, 2, 2, 2)
        override[..., 0, 0] = 1.0
        output, debug = self.stack(
            self.target,
            self.warp,
            self.payload(),
            self.spatial,
            self.spatial,
            2,
            2,
            2,
            gate_override=override,
        )
        self.assertTrue(debug["gate_override_used"])
        self.assertGreater(float((output - self.target).abs().max()), 0.0)
        self.assertEqual(float(debug["final_gate"][..., 1, 1].abs().max()), 0.0)

        _, disabled = self.stack(
            self.target,
            self.warp,
            self.payload(valid=0.0),
            self.spatial,
            self.spatial,
            2,
            2,
            2,
            gate_override=TORCH.ones_like(override),
        )
        self.assertEqual(float(disabled["final_gate"].abs().max()), 0.0)
        self.assertEqual(float(disabled["interaction_injection_map"].abs().max()), 0.0)

        temporal_target = TORCH.randn(1, 20, 16)
        temporal_warp = TORCH.randn(1, 20, 16)
        temporal_payload = self.payload()
        temporal_payload["frame_action_mask"] = TORCH.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        _, temporal_debug = self.stack(
            temporal_target,
            temporal_warp,
            temporal_payload,
            TORCH.ones(1, 1, 5, 2, 2),
            TORCH.ones(1, 1, 5, 2, 2),
            5,
            2,
            2,
            gate_override=TORCH.ones(1, 1, 5, 2, 2),
        )
        self.assertEqual(float(temporal_debug["final_gate"][:, :, 0].abs().max()), 0.0)
        self.assertEqual(float(temporal_debug["final_gate"][:, :, 2:].abs().max()), 0.0)

    def test_negative_raw_gate_has_gradient(self):
        payload = self.payload(valid=0.0)
        payload["frame_action_mask"].zero_()
        _, debug = self.stack(self.target, self.warp, payload, self.spatial, self.spatial, 2, 2, 2)
        debug["raw_gate"].mean().backward()
        grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.stack.router.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad, 0.0)

    def test_adapter_overfit_positive_actions_all_have_adapter_gradient(self):
        INTERACTION.configure_interaction_trainability(self.stack, "adapter_overfit")
        override = TORCH.ones(1, 1, 2, 2, 2)
        for action_id in (1, 2, 3):
            self.stack.zero_grad(set_to_none=True)
            payload = self.payload()
            payload["action_ids"] = TORCH.tensor([action_id])
            output, _ = self.stack(
                self.target,
                self.warp,
                payload,
                self.spatial,
                self.spatial,
                2,
                2,
                2,
                gate_override=override,
            )
            output.sum().backward()
            adapter_grad = sum(
                float(parameter.grad.abs().sum())
                for parameter in self.stack.adapter.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(adapter_grad, 0.0, f"action_id={action_id}")

    def test_inactive_stages_and_stage0_teacher_support(self):
        _, inactive = self.stack(
            self.target, self.warp, self.payload(), self.spatial, self.spatial, 2, 2, 2, stage_id=1
        )
        self.assertEqual(float(inactive["interaction_injection_map"].abs().max()), 0.0)
        teacher = TORCH.zeros(1, 1, 4, 8, 8)
        teacher[..., 3, 3] = 1.0
        aligned = INTERACTION.align_interaction_signals_to_grid(
            self.payload(),
            batch_size=1,
            temporal=2,
            height=2,
            width=2,
            device=self.target.device,
            visibility=self.spatial,
            world_valid=self.spatial,
            teacher=teacher,
        )
        self.assertGreater(float((aligned["teacher"] > 0).sum()), 0.0)

    def test_mine_progress_is_source_window_invariant(self):
        first = INTERACTION.mine_progress_for_source_frames(range(100, 110), 100, 120)
        second = INTERACTION.mine_progress_for_source_frames(range(105, 115), 100, 120)
        self.assertEqual(first[5:], second[:5])
        self.assertTrue(all(0.0 <= value < 1.0 for value in first + second))

    def test_interaction_and_optimizer_round_trip(self):
        groups = INTERACTION.configure_interaction_trainability(self.stack, "joint_pilot")
        parameters = [parameter for values in groups.values() for parameter in values]
        optimizer = TORCH.optim.AdamW(parameters, lr=1.0e-3)
        optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in parameters).backward()
        optimizer.step()
        model_state = self.stack.state_dict()
        optimizer_state = optimizer.state_dict()

        restored = INTERACTION.InteractionConditioningStack(16, semantic_dim=8, rank=4)
        restored_groups = INTERACTION.configure_interaction_trainability(restored, "joint_pilot")
        restored_parameters = [parameter for values in restored_groups.values() for parameter in values]
        restored_optimizer = TORCH.optim.AdamW(restored_parameters, lr=1.0e-3)
        restored.load_state_dict(model_state, strict=True)
        restored_optimizer.load_state_dict(optimizer_state)
        for name, value in self.stack.state_dict().items():
            self.assertTrue(TORCH.equal(value, restored.state_dict()[name]), name)
        self.assertEqual(len(optimizer.state), len(restored_optimizer.state))


if __name__ == "__main__":
    unittest.main()
