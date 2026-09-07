"""Pure imitation from multiple fixed depth-LoRA teachers.

This class is named PPO_WAQ_Distill to fit the repo's algorithm registry. 
It intentionally performs no PPO optimization:
student actions control the environment, while the selected LoRA teacher only
provides an action label for the same state.
"""

from __future__ import annotations


from itertools import chain
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.storage import RolloutStorageWAQDistill


class PPO_WAQ_Distill:
    def __init__(
        self,
        actor_critic,
        teachers,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        distill_target="l1",
        distillation_loss_coef=1.0,
        train_vae=True,
        train_visual_encoder=True,
        device="cpu",
        **unused_ppo_kwargs,
    ):
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)
        self.teachers = nn.ModuleList(teachers).to(self.device)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.max_grad_norm = float(max_grad_norm)
        self.distill_target = str(distill_target)
        self.distillation_loss_coef = float(
            distillation_loss_coef
        )

        if len(self.teachers) == 0:
            raise ValueError("PPO_WAQ_Distill needs at least one teacher.")

        # LoRA teachers are fixed action labelers.
        self.teachers.eval()
        for teacher in self.teachers:
            teacher.requires_grad_(False)

        # Pure imitation does not train PPO-only policy components.
        self.actor_critic.critic.requires_grad_(False)
        self.actor_critic.std.requires_grad_(False)

        trainable_groups = [self.actor_critic.actor.parameters()]
        if train_vae:
            trainable_groups.append(self.actor_critic.vae.parameters())
        else:
            self.actor_critic.vae.requires_grad_(False)

        if train_visual_encoder:
            trainable_groups.append(
                self.actor_critic.visual_encoder.parameters()
            )
        else:
            self.actor_critic.visual_encoder.requires_grad_(False)

        self.distillation_parameters = [
            parameter
            for parameter in chain.from_iterable(trainable_groups)
            if parameter.requires_grad
        ]
        if not self.distillation_parameters:
            raise ValueError("No trainable student parameters selected.")

        self.optimizer = optim.Adam(
            self.distillation_parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.storage = None
        self.transition = RolloutStorageWAQDistill.Transition()

        if unused_ppo_kwargs:
            print(
                "PPO_WAQ_Distill: ignoring inherited PPO-only keys: "
                + ", ".join(sorted(unused_ppo_kwargs))
            )

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        obs_history_shape,
        depth_image_shape,
        action_shape,
    ):
        self.storage = RolloutStorageWAQDistill(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            obs_history_shape=obs_history_shape,
            depth_image_shape=depth_image_shape,
            action_shape=action_shape,
            device=self.device,
        )

    @staticmethod
    def _pad_depth_images(depth_images, num_envs):
        """Match the repo's depth_actor behavior for non-camera envs.

        Genesis may provide images for only `num_camera_envs`. The existing
        depth actor gives remaining environments zero visual latents. We make
        that explicit before storing data so every sample remains aligned.
        """
        if depth_images.shape[0] == num_envs:
            return depth_images
        if depth_images.shape[0] > num_envs:
            raise ValueError("More depth images than environments.")

        padded = torch.zeros(
            num_envs,
            *depth_images.shape[1:],
            dtype=depth_images.dtype,
            device=depth_images.device,
        )
        padded[: depth_images.shape[0]].copy_(depth_images)
        return padded

    @torch.inference_mode()
    def _select_teacher_actions(
        self,
        observations,
        observation_histories,
        depth_images,
        teacher_ids,
    ):
        flat_ids = teacher_ids.view(-1).to(dtype=torch.long)
        if flat_ids.shape[0] != observations.shape[0]:
            raise ValueError("Expected one teacher ID per environment.")
        if torch.any(flat_ids < 0) or torch.any(
            flat_ids >= len(self.teachers)
        ):
            raise ValueError("Teacher ID is outside configured range.")

        selected = torch.empty(
            observations.shape[0],
            self.actor_critic.std.numel(),
            dtype=observations.dtype,
            device=observations.device,
        )

        # Compute each teacher on the full aligned batch, then select rows.
        # This avoids incorrect depth indexing when teacher masks are sparse.
        for teacher_index, teacher in enumerate(self.teachers):
            mask = flat_ids == teacher_index
            if not torch.any(mask):
                continue
            all_actions = teacher.act_inference(
                observations,
                observation_histories,
                depth_images,
            )
            selected[mask] = all_actions[mask]

        return selected

    @torch.inference_mode()
    def act(
        self,
        observations,
        observation_histories,
        depth_images,
        teacher_ids,
    ):
        aligned_depth = self._pad_depth_images(
            depth_images,
            observations.shape[0],
        )

        # The deterministic student action is the only action executed.
        student_actions = self.actor_critic.act_inference(
            observations,
            observation_histories,
            aligned_depth,
        )

        # Labels are generated from exactly the same state.
        teacher_actions = self._select_teacher_actions(
            observations,
            observation_histories,
            aligned_depth,
            teacher_ids,
        )

        self.transition.observations = observations
        self.transition.observation_histories = observation_histories
        self.transition.depth_images = aligned_depth
        self.transition.teacher_actions = teacher_actions
        self.transition.teacher_ids = teacher_ids
        return student_actions

    def process_env_step(self, *unused_args, **unused_kwargs):
        if self.storage is None:
            raise RuntimeError("Storage has not been initialized.")
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, *unused_args, **unused_kwargs):
        return None

    def _loss_per_sample(self, student_actions, teacher_actions):
        difference = student_actions - teacher_actions
        if self.distill_target == "l1":
            return difference.abs().sum(dim=-1)
        if self.distill_target == "mse":
            return difference.square().mean(dim=-1)
        if self.distill_target == "mse_sum":
            return difference.square().sum(dim=-1)
        if self.distill_target == "l2":
            return torch.linalg.vector_norm(
                difference, ord=2, dim=-1
            )
        raise ValueError(
            f"Unsupported distill_target={self.distill_target!r}."
        )

    def update(self) -> Tuple[float, Dict[str, float]]:
        if self.storage is None:
            raise RuntimeError("Storage has not been initialized.")

        total_loss = 0.0
        total_l1 = 0.0
        total_mse = 0.0
        update_count = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for batch in generator:
            student_actions = self.actor_critic.act_inference(
                batch["observations"],
                batch["observation_histories"],
                batch["depth_images"],
            )
            teacher_actions = batch["teacher_actions"]

            raw_loss = self._loss_per_sample(
                student_actions,
                teacher_actions,
            ).mean()
            loss = self.distillation_loss_coef * raw_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.distillation_parameters,
                self.max_grad_norm,
            )
            self.optimizer.step()

            with torch.no_grad():
                difference = student_actions - teacher_actions
                total_loss += raw_loss.item()
                total_l1 += difference.abs().mean().item()
                total_mse += difference.square().mean().item()
                update_count += 1

        if update_count == 0:
            raise RuntimeError("No distillation minibatches generated.")

        self.storage.clear()
        mean_loss = total_loss / update_count
        return mean_loss, {
            "distillation_loss": mean_loss,
            "action_l1_mean": total_l1 / update_count,
            "action_mse_mean": total_mse / update_count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "optimizer_updates": update_count,
        }
