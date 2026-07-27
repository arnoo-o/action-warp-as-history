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

    def forward(self, action_ids, block_ids, event_frames, total_frames, event_valid):
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
        return token * event_valid.to(token).unsqueeze(-1)


class InteractionRouter(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int = 256, rank: int = 64):
        super().__init__()
        rank = int(min(rank, hidden_dim))
        self.target_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.warp_projection = nn.Linear(hidden_dim, rank, bias=False)
        self.semantic_projection = nn.Linear(min(semantic_dim, hidden_dim), rank, bias=False)
        self.temporal_projection = nn.Sequential(
            nn.Linear(5, rank),
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
        frame_positions,
        event_positions,
        event_valid,
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
        return gate * visibility.to(gate) * event_valid.to(gate).view(-1, 1, 1)


class InteractionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int = 256, rank: int = 64, scale: float = 0.1):
        super().__init__()
        rank = int(min(rank, hidden_dim))
        self.target_down = nn.Linear(hidden_dim, rank, bias=False)
        self.warp_down = nn.Linear(hidden_dim, rank, bias=False)
        self.semantic_down = nn.Linear(min(semantic_dim, hidden_dim), rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.scale = float(scale)
        nn.init.zeros_(self.up.weight)

    def forward(self, target_tokens, warp_tokens, interaction_token, gate, stage_scale=1.0):
        projection_dtype = self.target_down.weight.dtype
        low_rank = (
            self.target_down(target_tokens.to(dtype=projection_dtype))
            + self.warp_down(warp_tokens.to(dtype=projection_dtype))
            + self.semantic_down(interaction_token.to(dtype=projection_dtype)).unsqueeze(1)
        )
        delta = self.up(F.silu(low_rank)).to(target_tokens)
        injection = self.scale * stage_scale.to(target_tokens) * gate.to(target_tokens) * delta
        return target_tokens + injection, injection


class InteractionConditioningStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        semantic_dim: int = 256,
        rank: int = 64,
        stage_warp_scales=DEFAULT_STAGE_WARP_SCALES,
        stage_adapter_scales=DEFAULT_STAGE_ADAPTER_SCALES,
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

    def forward(
        self,
        target_tokens,
        warp_tokens,
        payload,
        visibility,
        temporal,
        height,
        width,
        interaction_adapter_enabled=True,
        stage_id=0,
        previous_gate=None,
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
        semantic = self.semantic_encoder(action_ids, block_ids, event_frames, total_frames, event_valid)
        stage_ids = torch.full((batch_size,), stage_id, device=device, dtype=torch.long)
        semantic = semantic + self.stage_embedding(stage_ids) * event_valid.to(semantic).unsqueeze(-1)

        if visibility is None:
            visibility = torch.ones(batch_size, 1, temporal, height, width, device=device, dtype=torch.float32)
        else:
            visibility = visibility.to(device=device, dtype=torch.float32)
            visibility = F.interpolate(visibility, size=(temporal, height, width), mode="nearest")
        visibility_tokens = visibility.flatten(2).transpose(1, 2)
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
        gate = self.router(
            semantic,
            warp_tokens,
            target_tokens,
            visibility_tokens,
            frame_positions,
            event_positions,
            event_valid,
            previous_gate=previous_gate_tokens,
            previous_support=previous_support_tokens,
            is_refinement=stage_id > 0,
        )
        if bool(interaction_adapter_enabled):
            output, injection = self.adapter(
                target_tokens,
                warp_tokens,
                semantic,
                gate,
                stage_scale=self.stage_adapter_scales[stage_id],
            )
        else:
            output = target_tokens
            injection = torch.zeros_like(target_tokens)
        predicted_gate = gate.transpose(1, 2).reshape(batch_size, 1, temporal, height, width)
        injection_map = injection.float().square().mean(dim=-1, keepdim=True).sqrt().transpose(1, 2).reshape(
            batch_size, 1, temporal, height, width
        )
        debug = {
            "stage_id": stage_id,
            "interaction_token": semantic,
            "predicted_gate": predicted_gate,
            f"predicted_gate_stage{stage_id}": predicted_gate,
            "interaction_injection_map": injection_map,
            f"interaction_injection_map_stage{stage_id}": injection_map,
            "previous_gate": previous_gate,
            "stage_warp_scale": self.stage_warp_scales[stage_id],
            "stage_adapter_scale": self.stage_adapter_scales[stage_id],
        }
        return output, debug
