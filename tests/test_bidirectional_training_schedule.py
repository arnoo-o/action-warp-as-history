import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_warp_as_history_lora.py"
CORE_PATH = REPO_ROOT / "warp_as_history" / "training" / "core.py"
SCHEDULE_FUNCTIONS = {
    "training_total_steps",
    "training_stage_for_step",
    "should_compute_bidirectional_feedback",
}


def load_schedule_functions():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in SCHEDULE_FUNCTIONS
    ]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TRAIN_SCRIPT), "exec"), namespace)
    return namespace


SCHEDULE = load_schedule_functions()


class BidirectionalTrainingScheduleSmokeTest(unittest.TestCase):
    def test_base_only_total_and_stage(self):
        total = SCHEDULE["training_total_steps"](1500, 1500, False)
        self.assertEqual(total, 1500)
        self.assertEqual(
            {SCHEDULE["training_stage_for_step"](step, 1500, False) for step in (0, 1499)},
            {"base"},
        )

    def test_two_stage_transition_and_custom_lengths(self):
        total = SCHEDULE["training_total_steps"](2000, 800, True)
        self.assertEqual(total, 2800)
        self.assertEqual(SCHEDULE["training_stage_for_step"](1999, 2000, True), "base")
        self.assertEqual(SCHEDULE["training_stage_for_step"](2000, 2000, True), "bidirectional")
        self.assertEqual(SCHEDULE["training_stage_for_step"](2799, 2000, True), "bidirectional")

    def test_feedback_interval_starts_at_stage_two_zero(self):
        hits = [
            step
            for step in range(2, 7)
            if SCHEDULE["should_compute_bidirectional_feedback"](step, 2, True, 2)
        ]
        self.assertEqual(hits, [2, 4, 6])
        self.assertFalse(SCHEDULE["should_compute_bidirectional_feedback"](1, 2, True, 2))
        self.assertFalse(SCHEDULE["should_compute_bidirectional_feedback"](2, 2, False, 2))

    def test_core_uses_no_grad_and_adapter_only_switch(self):
        source = CORE_PATH.read_text(encoding="utf-8")
        self.assertIn("with torch.no_grad():", source)
        self.assertIn("interaction_adapter_enabled=False", source)
        self.assertIn("refine_interaction_teacher", source)

    def test_checkpoint_contains_stage_progress_and_teacher_cache(self):
        source = TRAIN_SCRIPT.read_text(encoding="utf-8")
        for field in (
            '"current_stage"',
            '"base_train_steps"',
            '"bidirectional_train_steps"',
            '"base_completed_steps"',
            '"bidirectional_completed_steps"',
            '"refined_teacher_cache"',
        ):
            self.assertIn(field, source)

    def test_requested_cli_defaults_are_stable(self):
        source = TRAIN_SCRIPT.read_text(encoding="utf-8")
        for declaration in (
            'parser.add_argument("--base_train_steps", type=int, default=1500)',
            'parser.add_argument("--bidirectional_train_steps", type=int, default=1500)',
            'parser.add_argument("--bidirectional_interval", type=int, default=8)',
            'parser.add_argument("--bidirectional_feedback_weight", type=float, default=0.5)',
            'parser.add_argument("--bidirectional_teacher_floor", type=float, default=0.5)',
        ):
            self.assertIn(declaration, source)
        self.assertIn('action=argparse.BooleanOptionalAction,\n        default=False,', source)


if __name__ == "__main__":
    unittest.main()
