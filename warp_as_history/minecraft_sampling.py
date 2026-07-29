"""Pure-Python deterministic sampling utilities for Minecraft training."""
from __future__ import annotations

import random


class StepCategorySampler:
    """Quota-exact step sampler with deterministic per-category row order."""

    def __init__(self, pools, ratios, total_steps, seed):
        self.pools = {name: list(values) for name, values in pools.items() if values}
        if not self.pools:
            raise ValueError("All category pools are empty.")
        self.requested_ratios = {name: max(float(value), 0.0) for name, value in ratios.items()}
        raw = {name: self.requested_ratios.get(name, 0.0) for name in self.pools}
        total = sum(raw.values())
        if total <= 0:
            raw = {name: 1.0 for name in self.pools}
            total = float(len(raw))
        self.ratios = {name: value / total for name, value in raw.items()}
        self.total_steps = int(total_steps)
        self.targets = self._integer_targets()
        self.schedule = self._build_schedule()
        self.occurrence_before = []
        seen = {name: 0 for name in self.pools}
        for category in self.schedule:
            self.occurrence_before.append(seen[category])
            seen[category] += 1
        self.orders = {}
        for offset, (name, pool) in enumerate(sorted(self.pools.items())):
            order = list(pool)
            random.Random(int(seed) + offset * 1009).shuffle(order)
            self.orders[name] = order

    def _integer_targets(self):
        raw = {name: self.ratios[name] * self.total_steps for name in self.pools}
        targets = {name: int(value) for name, value in raw.items()}
        remaining = self.total_steps - sum(targets.values())
        priority = sorted(
            raw,
            key=lambda item: (raw[item] - targets[item], item),
            reverse=True,
        )
        for name in priority[:remaining]:
            targets[name] += 1
        return targets

    def _build_schedule(self):
        emitted = {name: 0 for name in self.pools}
        schedule = []
        for step in range(self.total_steps):
            candidates = [name for name in self.pools if emitted[name] < self.targets[name]]
            name = max(
                candidates,
                key=lambda item: (
                    (step + 1) * self.targets[item] / max(self.total_steps, 1) - emitted[item],
                    item,
                ),
            )
            emitted[name] += 1
            schedule.append(name)
        return schedule

    def sample(self, step):
        step = int(step)
        category = self.schedule[step]
        return category, self.sample_category(category, self.occurrence_before[step])

    def sample_category(self, category, occurrence):
        return self.orders[category][int(occurrence) % len(self.orders[category])]

    def report(self, completed_steps):
        completed = min(max(int(completed_steps), 0), self.total_steps)
        counts = {name: 0 for name in self.pools}
        for name in self.schedule[:completed]:
            counts[name] += 1
        return {
            "requested_ratios": self.requested_ratios,
            "effective_ratios": self.ratios,
            "target_steps": self.targets,
            "actual_steps": counts,
            "actual_ratios": {name: value / max(completed, 1) for name, value in counts.items()},
        }


def build_interaction_event_window(
    event_frame,
    *,
    num_source_frames,
    window_size,
    rng,
    local_min,
    local_max,
    require_later=True,
):
    if int(num_source_frames) < int(window_size):
        raise ValueError("Video is shorter than one interaction training window.")
    local_min = max(1, int(local_min))
    local_max = min(int(local_max), int(window_size) - 2)
    if local_min > local_max:
        raise ValueError("interaction_event_local_min/max leave no post-event frames.")
    preferred_local = rng.randint(local_min, local_max)
    start = min(
        max(int(event_frame) - preferred_local, 0),
        int(num_source_frames) - int(window_size),
    )
    local = int(event_frame) - start
    if not (local_min <= local <= local_max):
        raise ValueError(
            f"Cannot place event frame {event_frame} in local range "
            f"[{local_min}, {local_max}] for {num_source_frames} frames."
        )
    if bool(require_later) and start <= 0:
        raise ValueError(f"Interaction event {event_frame} cannot provide a later-chunk history baseline.")
    indices = list(range(start, start + int(window_size)))
    if int(event_frame) not in indices:
        raise AssertionError("Interaction window lost its positive event.")
    return indices, int(local)


def interaction_payload_for_chunk(payload, *, chunk_index, window_frames, consumed_event_frames=()):
    """Route one global event to exactly one autoregressive chunk."""
    if payload is None:
        return None
    global_event_frame = int(payload.get("event_frame", -1))
    consumed = {int(value) for value in consumed_event_frames}
    start = int(chunk_index) * int(window_frames)
    end = start + int(window_frames)
    if global_event_frame in consumed or not (start <= global_event_frame < end):
        return None
    routed = dict(payload)
    routed["global_event_frame"] = global_event_frame
    routed["event_frame"] = global_event_frame - start
    routed["event_valid"] = float(payload.get("event_valid", 1.0))
    return routed


def interaction_event_identity(payload):
    if payload is None:
        return None
    explicit = str(payload.get("event_id", "") or "").strip()
    if explicit:
        return explicit
    return (
        f"{int(payload.get('event_frame', -1))}:"
        f"{payload.get('action_type', 'none')}:"
        f"{payload.get('block_id', payload.get('object_id'))}"
    )
