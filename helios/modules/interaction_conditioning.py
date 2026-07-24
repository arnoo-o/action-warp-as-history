from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


INTERACTION_ACTIONS = ("none", "place", "mine_active", "mine_complete", "primary_fire")
INTERACTION_ACTION_TO_ID = {name: index for index, name in enumerate(INTERACTION_ACTIONS)}
INTERACTION_BLOCK_BUCKETS = 4096


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
        time_embedding = self.time_projection(sinusoidal_scalar(event_position, self.semantic_dim))
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
        return token * event_valid.float().unsqueeze(-1)


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
        )
        routed = (
            self.target_projection(target_tokens)
            + self.warp_projection(warp_tokens)
            + self.semantic_projection(interaction_token).unsqueeze(1)
            + self.temporal_projection(temporal_features)
        )
        gate = torch.sigmoid(self.output(F.silu(routed)))
        return gate * visibility.float() * event_valid.float().view(-1, 1, 1)


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

    def forward(self, target_tokens, warp_tokens, interaction_token, gate):
        low_rank = (
            self.target_down(target_tokens)
            + self.warp_down(warp_tokens)
            + self.semantic_down(interaction_token).unsqueeze(1)
        )
        delta = self.up(F.silu(low_rank)).to(target_tokens)
        injection = self.scale * gate.to(target_tokens) * delta
        return target_tokens + injection, injection


class InteractionConditioningStack(nn.Module):
    def __init__(self, hidden_dim: int, semantic_dim: int = 256, rank: int = 64):
        super().__init__()
        self.semantic_encoder = InteractionSemanticEncoder(hidden_dim, semantic_dim=semantic_dim)
        self.router = InteractionRouter(hidden_dim, semantic_dim=semantic_dim, rank=rank)
        self.adapter = InteractionAdapter(hidden_dim, semantic_dim=semantic_dim, rank=rank)

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
        semantic = self.semantic_encoder(action_ids, block_ids, event_frames, total_frames, event_valid)

        if visibility is None:
            visibility = torch.ones(batch_size, 1, temporal, height, width, device=device, dtype=torch.float32)
        else:
            visibility = visibility.to(device=device, dtype=torch.float32)
            visibility = F.interpolate(visibility, size=(temporal, height, width), mode="nearest")
        visibility_tokens = visibility.flatten(2).transpose(1, 2)
        frame_axis = torch.linspace(0.0, 1.0, temporal, device=device, dtype=torch.float32)
        frame_positions = frame_axis.repeat_interleave(height * width).unsqueeze(0).expand(batch_size, -1)
        event_positions = event_frames / (total_frames - 1.0).clamp_min(1.0)
        gate = self.router(
            semantic,
            warp_tokens,
            target_tokens,
            visibility_tokens,
            frame_positions,
            event_positions,
            event_valid,
        )
        if bool(interaction_adapter_enabled):
            output, injection = self.adapter(target_tokens, warp_tokens, semantic, gate)
        else:
            output = target_tokens
            injection = torch.zeros_like(target_tokens)
        debug = {
            "interaction_token": semantic,
            "predicted_gate": gate.transpose(1, 2).reshape(batch_size, 1, temporal, height, width),
            "interaction_injection_map": injection.float().square().mean(dim=-1, keepdim=True).sqrt()
            .transpose(1, 2)
            .reshape(batch_size, 1, temporal, height, width),
        }
        return output, debug
