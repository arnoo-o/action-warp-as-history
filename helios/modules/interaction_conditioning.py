from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


INTERACTION_ACTIONS = ("none", "place", "mine_active", "mine_complete", "primary_fire")
INTERACTION_ACTION_TO_ID = {name: index for index, name in enumerate(INTERACTION_ACTIONS)}
INTERACTION_BLOCK_BUCKETS = 4096
INTERACTION_PYRAMID_STAGES = 3
DEFAULT_STAGE_WARP_SCALES = (1.0, 0.5, 0.25)
DEFAULT_STAGE_ADAPTER_SCALES = (1.0, 0.5, 0.25)


def _as_bcthw(value, batch_size, device, *, default_shape=None):
    if value is None:
        if default_shape is None:
            return None
        return torch.ones(batch_size, 1, *default_shape, device=device, dtype=torch.float32)
    value = torch.as_tensor(value, device=device, dtype=torch.float32)
    if value.ndim == 4:
        value = value.unsqueeze(1)
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, -1, -1, -1, -1)
    if value.ndim != 5 or value.shape[0] != batch_size:
        raise ValueError(f"interaction map must have shape [B,1,T,H,W], got {tuple(value.shape)}")
    return value


def pool_frame_signals_to_time(frame_action_mask, frame_progress_curve, temporal):
    """Coverage-pool source-frame controls onto an arbitrary latent/token time axis."""
    frame_count = int(frame_action_mask.shape[1])
    pooled_mask = []
    pooled_progress = []
    for time_index in range(int(temporal)):
        start = int(math.floor(time_index * frame_count / int(temporal)))
        end = max(int(math.ceil((time_index + 1) * frame_count / int(temporal))), start + 1)
        mask_slice = frame_action_mask[:, start:end]
        progress_slice = frame_progress_curve[:, start:end]
        coverage = mask_slice.amax(dim=1)
        progress = (progress_slice * mask_slice).sum(dim=1) / mask_slice.sum(dim=1).clamp_min(1.0)
        pooled_mask.append(coverage)
        pooled_progress.append(progress * (coverage > 0).to(progress))
    return torch.stack(pooled_mask, dim=1), torch.stack(pooled_progress, dim=1)


def mine_progress_for_source_frames(source_frames, action_start_frame, complete_frame):
    """Return source-timeline progress so overlapping windows agree exactly."""
    start = int(action_start_frame)
    complete = int(complete_frame)
    duration = max(complete - start, 1)
    return [
        min(max((int(frame) - start) / float(duration), 0.0), 1.0 - 1.0e-6)
        if start <= int(frame) < complete
        else 0.0
        for frame in source_frames
    ]


def align_interaction_signals_to_grid(
    payload,
    *,
    batch_size,
    temporal,
    height,
    width,
    device,
    visibility=None,
    world_valid=None,
    teacher=None,
):
    """Share RGB-frame -> latent/token-grid alignment across Router, Adapter and loss."""
    frame_action = InteractionConditioningStack._payload_frame_signal(
        payload, "frame_action_mask", batch_size, device
    )
    frame_progress = InteractionConditioningStack._payload_frame_signal(
        payload, "frame_progress_curve", batch_size, device
    )
    action_time, progress_time = pool_frame_signals_to_time(frame_action, frame_progress, temporal)
    action = action_time.view(batch_size, 1, temporal, 1, 1).expand(-1, 1, -1, height, width)
    progress = progress_time.view(batch_size, 1, temporal, 1, 1).expand(-1, 1, -1, height, width)

    visibility = _as_bcthw(
        visibility, batch_size, device, default_shape=(temporal, height, width)
    )
    visibility = F.adaptive_avg_pool3d(visibility, (temporal, height, width)).clamp(0.0, 1.0)
    world_valid = _as_bcthw(
        world_valid, batch_size, device, default_shape=(temporal, height, width)
    )
    world_valid = (
        F.adaptive_avg_pool3d(world_valid, (temporal, height, width)) >= 0.999
    ).to(dtype=torch.float32)

    aligned = {
        "action": action,
        "progress": progress,
        "visibility": visibility,
        "world_valid": world_valid,
        "valid": action * visibility * world_valid,
    }
    if teacher is not None:
        teacher = _as_bcthw(teacher, batch_size, device)
        # Adaptive max pooling preserves compact interaction regions on the coarse Stage 0 grid.
        aligned["teacher"] = F.adaptive_max_pool3d(
            teacher, (temporal, height, width)
        ).clamp(0.0, 1.0)
    return aligned


def configure_interaction_trainability(stack, mode):
    """Apply the four module-validation freeze policies and return named optimizer groups."""
    mode = str(mode)
    if mode not in {"router_overfit", "adapter_overfit", "joint_pilot", "joint_stage0"}:
        raise ValueError(f"Unsupported interaction training mode: {mode}")
    groups = {"interaction_semantic": [], "interaction_router": [], "interaction_adapter": []}
    for name, parameter in stack.named_parameters():
        if name.startswith(("stage_warp_scales", "stage_adapter_scales")):
            trainable = False
        elif mode == "router_overfit":
            trainable = name.startswith(("semantic_encoder.", "stage_embedding.", "router."))
        elif mode == "adapter_overfit":
            trainable = name.startswith("adapter.")
        else:
            trainable = name.startswith(("semantic_encoder.", "stage_embedding.", "router.", "adapter."))
        parameter.requires_grad_(trainable)
        if not trainable:
            continue
        if name.startswith("adapter."):
            groups["interaction_adapter"].append(parameter)
        elif name.startswith("router."):
            groups["interaction_router"].append(parameter)
        else:
            groups["interaction_semantic"].append(parameter)
    return groups


def interaction_action_id(action_type: str | None) -> int:
    return int(INTERACTION_ACTION_TO_ID.get(str(action_type or "none").strip().lower(), 0))


def interaction_block_id(value: str | int | None, buckets: int = INTERACTION_BLOCK_BUCKETS) -> int:
    if value is None or str(value).strip() == "":
        return 0
    if isinstance(value, int):
        return 1 + (abs(int(value)) % max(int(buckets) - 1, 1))
    digest = hashlib.sha256(str(value).strip().lower().encode("utf-8")).digest()
    return 1 + (int.from_bytes(digest[:8], "little") % max(int(buckets) - 1, 1))


def sinusoidal_scalar(value: torch.Tensor, dim: int) -> torch.Tensor:
    half = max(int(dim) // 2, 1)
    frequencies = torch.exp(
        torch.arange(half, device=value.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = value.float().unsqueeze(-1) * frequencies
    encoded = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if encoded.shape[-1] < int(dim):
        encoded = F.pad(encoded, (0, int(dim) - encoded.shape[-1]))
    return encoded[..., : int(dim)]


class InteractionSemanticEncoder(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int = 256, block_buckets: int = INTERACTION_BLOCK_BUCKETS):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(min(semantic_dim, hidden_dim))
        self.action_embedding = nn.Embedding(len(INTERACTION_ACTIONS), self.semantic_dim)
        self.block_embedding = nn.Embedding(int(block_buckets), self.semantic_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(self.semantic_dim, self.semantic_dim),
            nn.SiLU(),
            nn.Linear(self.semantic_dim, self.semantic_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.semantic_dim * 3, self.semantic_dim),
            nn.SiLU(),
            nn.LayerNorm(self.semantic_dim),
        )

    def forward(self, action_ids, block_ids, event_frames, total_frames):
        denominator = (total_frames.float() - 1.0).clamp_min(1.0)
        event_position = event_frames.float() / denominator
        module_dtype = self.action_embedding.weight.dtype
        time_embedding = self.time_projection(
            sinusoidal_scalar(event_position, self.semantic_dim).to(dtype=module_dtype)
        )
        token = self.fusion(
            torch.cat(
                [
                    self.action_embedding(action_ids.long()),
                    self.block_embedding(block_ids.long()),
                    time_embedding,
                ],
                dim=-1,
            )
        )
        return token


class InteractionRouter(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int = 256, rank: int = 64):
        super().__init__()
        rank = int(min(rank, hidden_dim))
        self.target_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.warp_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.semantic_projection = nn.Linear(min(semantic_dim, hidden_dim), rank, bias=False)
        self.temporal_projection = nn.Sequential(
            nn.Linear(7, rank),
            nn.SiLU(),
            nn.Linear(rank, rank),
        )
        self.output = nn.Linear(rank, 1)

    def forward(
        self,
        interaction_token,
        warp_tokens,
        target_tokens,
        visibility,
        action_mask,
        progress,
        frame_positions,
        event_positions,
        previous_gate=None,
        previous_support=None,
        is_refinement=False,
    ):
        relative = frame_positions.float() - event_positions.float().unsqueeze(1)
        temporal_features = torch.stack(
            [
                frame_positions.float(),
                relative,
                torch.sin(math.pi * relative),
                torch.cos(math.pi * relative),
                visibility.squeeze(-1).float(),
                action_mask.squeeze(-1).float(),
                progress.squeeze(-1).float(),
            ],
            dim=-1,
        ).to(dtype=self.temporal_projection[0].weight.dtype)
        projection_dtype = self.target_projection.weight.dtype
        target_for_projection = target_tokens.to(dtype=projection_dtype)
        warp_for_projection = warp_tokens.to(dtype=projection_dtype)
        semantic_for_projection = interaction_token.to(dtype=projection_dtype)
        routed = (
            self.target_projection(target_for_projection)
            + self.warp_projection(warp_for_projection)
            + self.semantic_projection(semantic_for_projection).unsqueeze(1)
            + self.temporal_projection(temporal_features)
        )
        logits = self.output(F.silu(routed))
        if bool(is_refinement):
            if previous_gate is None:
                raise ValueError("coarse-to-fine interaction refinement requires previous_gate.")
            previous_gate = previous_gate.to(device=logits.device, dtype=logits.dtype)
            if previous_support is None:
                raise ValueError("coarse-to-fine interaction refinement requires previous_support.")
            delta = 0.25 * torch.tanh(logits) * previous_support.to(logits)
            gate = (previous_gate + delta).clamp(0.0, 1.0)
        else:
            gate = torch.sigmoid(logits)
        return gate


class InteractionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int = 256, rank: int = 64, scale: float = 1.0):
        super().__init__()
        rank = int(min(rank, hidden_dim))
        self.target_down = nn.Linear(hidden_dim, rank, bias=False)
        self.warp_down = nn.Linear(hidden_dim, rank, bias=False)
        self.semantic_down = nn.Linear(min(semantic_dim, hidden_dim), rank, bias=False)
        self.progress_down = nn.Linear(1, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.scale = float(scale)
        nn.init.zeros_(self.progress_down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, target_tokens, warp_tokens, interaction_token, progress, gate, stage_scale=1.0):
        projection_dtype = self.target_down.weight.dtype
        low_rank = (
            self.target_down(target_tokens.to(dtype=projection_dtype))
            + self.warp_down(warp_tokens.to(dtype=projection_dtype))
            + self.semantic_down(interaction_token.to(dtype=projection_dtype)).unsqueeze(1)
            + self.progress_down(progress.to(dtype=projection_dtype))
        )
        delta = self.up(F.silu(low_rank)).to(target_tokens)
        injection = self.scale * stage_scale.to(target_tokens) * gate.to(target_tokens) * delta
        return target_tokens + injection, delta, injection


class InteractionConditioningStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        semantic_dim: int = 256,
        rank: int = 64,
        stage_warp_scales=DEFAULT_STAGE_WARP_SCALES,
        stage_adapter_scales=DEFAULT_STAGE_ADAPTER_SCALES,
        active_stages=(0,),
    ):
        super().__init__()
        self.semantic_encoder = InteractionSemanticEncoder(hidden_dim, semantic_dim=semantic_dim)
        self.stage_embedding = nn.Embedding(INTERACTION_PYRAMID_STAGES, self.semantic_encoder.semantic_dim)
        nn.init.zeros_(self.stage_embedding.weight)
        self.router = InteractionRouter(hidden_dim, semantic_dim=semantic_dim, rank=rank)
        self.adapter = InteractionAdapter(hidden_dim, semantic_dim=semantic_dim, rank=rank)
        self.stage_warp_scales = nn.Parameter(self._stage_scale_tensor(stage_warp_scales, "stage_warp_scales"))
        self.stage_adapter_scales = nn.Parameter(
            self._stage_scale_tensor(stage_adapter_scales, "stage_adapter_scales")
        )
        self.active_stages = tuple(sorted({int(stage) for stage in active_stages}))
        if not self.active_stages or any(stage < 0 or stage >= INTERACTION_PYRAMID_STAGES for stage in self.active_stages):
            raise ValueError(f"active_stages must be a non-empty subset of [0, 1, 2], got {self.active_stages}.")
        self.stage_warp_scales.requires_grad_(False)
        self.stage_adapter_scales.requires_grad_(False)

    @staticmethod
    def _stage_scale_tensor(values, name):
        values = tuple(float(value) for value in values)
        if len(values) != INTERACTION_PYRAMID_STAGES:
            raise ValueError(f"{name} must contain {INTERACTION_PYRAMID_STAGES} values, got {values}.")
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _payload_tensor(payload, name, batch_size, device, dtype, default):
        value = payload.get(name, default)
        value = torch.as_tensor(value, device=device, dtype=dtype).flatten()
        if value.numel() == 1 and batch_size > 1:
            value = value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(f"interaction payload {name} must have batch size {batch_size}, got {value.numel()}.")
        return value

    @staticmethod
    def _payload_frame_signal(payload, name, batch_size, device, default=0.0):
        value = payload.get(name)
        if value is None:
            total = torch.as_tensor(payload.get("total_frames", [1]), device=device).flatten()
            frame_count = max(int(total.max().item()), 1)
            return torch.full((batch_size, frame_count), float(default), device=device, dtype=torch.float32)
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise ValueError(f"interaction payload {name} must have shape [B,F], got {tuple(value.shape)}.")
        return value

    def forward(
        self,
        target_tokens,
        warp_tokens,
        payload,
        visibility,
        world_valid,
        temporal,
        height,
        width,
        interaction_adapter_enabled=True,
        stage_id=0,
        previous_gate=None,
        gate_override=None,
    ):
        batch_size = target_tokens.shape[0]
        device = target_tokens.device
        action_ids = self._payload_tensor(payload, "action_ids", batch_size, device, torch.long, 0)
        block_ids = self._payload_tensor(payload, "block_ids", batch_size, device, torch.long, 0)
        event_frames = self._payload_tensor(payload, "event_frames", batch_size, device, torch.float32, 0.0)
        total_frames = self._payload_tensor(
            payload, "total_frames", batch_size, device, torch.float32, max(int(temporal), 1)
        )
        event_valid = self._payload_tensor(payload, "event_valid", batch_size, device, torch.float32, 0.0)
        stage_id = int(stage_id)
        if stage_id < 0 or stage_id >= INTERACTION_PYRAMID_STAGES:
            raise ValueError(f"interaction stage_id must be in [0, {INTERACTION_PYRAMID_STAGES - 1}].")
        if stage_id not in self.active_stages:
            zero_gate = torch.zeros(batch_size, 1, temporal, height, width, device=device, dtype=target_tokens.dtype)
            return target_tokens, {
                "stage_id": stage_id,
                "interaction_active": False,
                "raw_gate": zero_gate,
                "final_gate": zero_gate,
                "valid_injection_region": zero_gate,
                "world_valid_region": zero_gate,
                "predicted_gate": zero_gate,
                f"predicted_gate_stage{stage_id}": zero_gate,
                "raw_delta_map": zero_gate,
                "interaction_injection_map": zero_gate,
                f"interaction_injection_map_stage{stage_id}": zero_gate,
                "previous_gate": None,
                "stage_warp_scale": self.stage_warp_scales[stage_id].detach(),
                "stage_adapter_scale": torch.ones((), device=device, dtype=torch.float32),
            }
        semantic = self.semantic_encoder(action_ids, block_ids, event_frames, total_frames)
        stage_ids = torch.full((batch_size,), stage_id, device=device, dtype=torch.long)
        semantic = semantic + self.stage_embedding(stage_ids)

        aligned = align_interaction_signals_to_grid(
            payload,
            batch_size=batch_size,
            temporal=temporal,
            height=height,
            width=width,
            device=device,
            visibility=visibility,
            world_valid=world_valid,
            teacher=gate_override,
        )
        visibility = aligned["visibility"]
        world_valid = aligned["world_valid"]
        visibility_tokens = visibility.flatten(2).transpose(1, 2)
        world_valid_tokens = world_valid.flatten(2).transpose(1, 2)
        temporal_action = aligned["action"][:, 0, :, 0, 0]
        temporal_progress = aligned["progress"][:, 0, :, 0, 0]
        action_tokens = temporal_action.repeat_interleave(height * width, dim=1).unsqueeze(-1)
        progress_tokens = temporal_progress.repeat_interleave(height * width, dim=1).unsqueeze(-1)
        warp_tokens = warp_tokens * self.stage_warp_scales[stage_id].to(warp_tokens)
        frame_axis = torch.linspace(0.0, 1.0, temporal, device=device, dtype=torch.float32)
        frame_positions = frame_axis.repeat_interleave(height * width).unsqueeze(0).expand(batch_size, -1)
        event_positions = event_frames / (total_frames - 1.0).clamp_min(1.0)
        previous_gate_tokens = None
        previous_support_tokens = None
        if stage_id > 0:
            if previous_gate is None:
                raise ValueError(f"interaction stage {stage_id} requires the previous stage predicted_gate.")
            previous_gate = F.interpolate(
                previous_gate.to(device=device, dtype=torch.float32),
                size=(temporal, height, width),
                mode="trilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            previous_gate_tokens = previous_gate.flatten(2).transpose(1, 2)
            previous_support_tokens = F.max_pool3d(
                previous_gate,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
            ).flatten(2).transpose(1, 2)
        raw_gate = self.router(
            semantic,
            warp_tokens,
            target_tokens,
            visibility_tokens,
            action_tokens,
            progress_tokens,
            frame_positions,
            event_positions,
            previous_gate=previous_gate_tokens,
            previous_support=previous_support_tokens,
            is_refinement=stage_id > 0,
        )
        injection_gate = raw_gate
        if gate_override is not None:
            injection_gate = aligned["teacher"].flatten(2).transpose(1, 2).detach().to(raw_gate)
        final_gate = (
            injection_gate
            * visibility_tokens.to(raw_gate)
            * world_valid_tokens.to(raw_gate)
            * action_tokens.to(raw_gate)
            * event_valid.to(raw_gate).view(-1, 1, 1)
        )
        valid_injection_tokens = (
            visibility_tokens.to(raw_gate)
            * world_valid_tokens.to(raw_gate)
            * action_tokens.to(raw_gate)
        )
        if bool(interaction_adapter_enabled):
            output, raw_delta, injection = self.adapter(
                target_tokens,
                warp_tokens,
                semantic,
                progress_tokens,
                final_gate,
                stage_scale=torch.ones((), device=device, dtype=torch.float32),
            )
        else:
            output = target_tokens
            raw_delta = torch.zeros_like(target_tokens)
            injection = torch.zeros_like(target_tokens)
        raw_gate_map = raw_gate.transpose(1, 2).reshape(batch_size, 1, temporal, height, width)
        final_gate_map = final_gate.transpose(1, 2).reshape(batch_size, 1, temporal, height, width)
        valid_injection_map = valid_injection_tokens.transpose(1, 2).reshape(
            batch_size, 1, temporal, height, width
        )
        world_valid_map = world_valid_tokens.transpose(1, 2).reshape(
            batch_size, 1, temporal, height, width
        )
        raw_delta_map = raw_delta.float().square().mean(dim=-1, keepdim=True).sqrt().transpose(1, 2).reshape(
            batch_size, 1, temporal, height, width
        )
        injection_map = injection.float().square().mean(dim=-1, keepdim=True).sqrt().transpose(1, 2).reshape(
            batch_size, 1, temporal, height, width
        )
        debug = {
            "stage_id": stage_id,
            "interaction_active": True,
            "interaction_token": semantic,
            "raw_gate": raw_gate_map,
            "gate_override_used": bool(gate_override is not None),
            "final_gate": final_gate_map,
            "valid_injection_region": valid_injection_map,
            "world_valid_region": world_valid_map,
            "predicted_gate": final_gate_map,
            f"predicted_gate_stage{stage_id}": final_gate_map,
            "raw_delta_map": raw_delta_map,
            "interaction_injection_map": injection_map,
            f"interaction_injection_map_stage{stage_id}": injection_map,
            "previous_gate": previous_gate,
            "stage_warp_scale": self.stage_warp_scales[stage_id],
            "stage_adapter_scale": torch.ones((), device=device, dtype=torch.float32),
        }
        return output, debug
