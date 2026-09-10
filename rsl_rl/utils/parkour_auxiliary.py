"""Disposable geometry heads; deliberately NOT part of the actor-critic state dict."""

import torch
from torch import nn
from torch.nn import functional as F


TARGET_NAMES = (
    "stairs_com_distance", "stairs_fl_distance", "stairs_fr_distance",
    "stairs_signed_step_height",
    "pit_com_distance", "pit_fl_distance", "pit_fr_distance", "pit_signed_height",
    "gap_com_distance", "gap_fl_distance", "gap_fr_distance", "gap_width",
)


class ParkourAuxiliary:
    def __init__(self, alg, cfg):
        self.cfg = dict(cfg)
        self.alg = alg
        model = alg.actor_critic
        size = model.vae.num_latent_dims + model.vae.num_explicit_dims + model.visual_encoder._output_size
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(size, cfg["hidden_dim"]), nn.ELU(), nn.Linear(cfg["hidden_dim"], out))
            for name, out in (("stairs", 4), ("pit", 4), ("gap", 4))
        }).to(alg.device)
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=cfg["learning_rate"])
        storage = alg.storage
        self.targets = torch.zeros(*storage.observation_histories.shape[:2], 12, device=alg.device)
        self.masks = torch.zeros_like(self.targets, dtype=torch.bool)
        self.ready = torch.zeros(storage.num_envs, device=alg.device, dtype=torch.bool)
        self.scales = torch.tensor([cfg["max_distance"]] * 3 + [0.3] +
                                   [cfg["max_distance"]] * 3 + [1.] + [cfg["max_distance"]] * 3 + [1.], device=alg.device)
        self.last_loss = 0.
        self.batch_active = False

    def capture(self, targets, masks):
        # Called before act()/env.step(); never store a view into environment state.
        step = self.alg.storage.step
        self.targets[step].copy_(targets)
        self.masks[step].copy_(masks & self.ready[:, None])

    def predict(self, history, depth):
        # Deterministic diagnostic decode; training reuses the existing VAE sample.
        z, _, velocity, _ = self.alg.actor_critic.vae.encode(history)
        return self.decode(z, velocity, depth)

    def decode(self, z, velocity, depth):
        visual = self.alg.actor_critic.visual_encoder(depth)
        context = torch.cat((z, velocity, visual), dim=-1)
        return torch.cat([self.heads[name](context) for name in ("stairs", "pit", "gap")], dim=-1)

    def loss(self, prediction, target, mask):
        regression = F.smooth_l1_loss(prediction / self.scales, target / self.scales, reduction="none")
        return (regression * mask).sum() / mask.sum().clamp_min(1)

    def reconstruction_loss(self, z, velocity, depth, terminated):
        # The same permutation as existing PPO/VAE/depth minibatches.
        idx = self.alg.storage.current_batch_indices
        mask = self.masks.flatten(0, 1)[idx] & terminated.bool()
        self.batch_active = bool(mask.any())
        if not self.batch_active:
            return z.new_zeros(())
        loss = self.loss(self.decode(z, velocity, depth), self.targets.flatten(0, 1)[idx], mask)
        self.last_loss += loss.detach().item()
        return self.cfg["loss_weight"] * loss

    def save(self, path):
        torch.save({"heads": self.heads.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "vae_optimizer": self.alg.vae_optimizer.state_dict(), "config": self.cfg,
                    "target_names": TARGET_NAMES, "distance_reference": "whole_robot_com_FL_FR",
                    "distance_direction": "horizontal_body_forward", "target_time": "action_time"}, path)

    def load(self, path, load_optimizer=True):
        state = torch.load(path, map_location=self.alg.device, weights_only=False)
        if state["config"] != self.cfg or tuple(state["target_names"]) != TARGET_NAMES:
            raise ValueError("Auxiliary sidecar configuration does not match this run")
        self.heads.load_state_dict(state["heads"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            self.alg.vae_optimizer.load_state_dict(state["vae_optimizer"])
