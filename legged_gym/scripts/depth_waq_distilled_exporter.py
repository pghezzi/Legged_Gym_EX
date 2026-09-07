# Isaac Gym must initialize before importing PyTorch.
from legged_gym import LEGGED_GYM_ROOT_DIR

import copy
import os
from datetime import datetime

import torch

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class DistilledDepthCNNExporter(torch.nn.Module):
    """Depth encoder half of the single distilled policy."""

    @torch.jit.unused
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual_encoder = copy.deepcopy(visual_encoder)

    @torch.jit.export
    def forward(self, depth_image):
        return self.visual_encoder(depth_image)

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        torch.jit.script(self).save(filename)


class DistilledFeaturesWaQExporter(torch.nn.Module):
    """Actor/VAE half of the single distilled policy."""

    @torch.jit.unused
    def __init__(self, actor, vae):
        super().__init__()
        self.actor = copy.deepcopy(actor)
        self.vae = copy.deepcopy(vae)

    @torch.jit.export
    def forward(self, observations, obs_history, visual_latent):
        mean_out = self.vae.inference(obs_history)
        return self.actor(torch.cat((observations, mean_out, visual_latent), dim=-1))

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        torch.jit.script(self).save(filename)


class DistilledDepthWaQExporter(torch.nn.Module):
    """Standalone TorchScript exporter for one distilled depth-DreamWaQ policy."""

    @torch.jit.unused
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
        self.visual_encoder = copy.deepcopy(actor_critic.visual_encoder)

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filename)

    @torch.jit.export
    def forward(
        self,
        observations,
        obs_history,
        depth_image,
    ):
        mean_out = self.vae.inference(obs_history)
        visual_latent = self.visual_encoder(depth_image)
        return self.actor(
            torch.cat(
                (observations, mean_out, visual_latent),
                dim=-1,
            )
        )

    @torch.jit.unused
    def split(self):
        """Return standalone CNN and actor/VAE modules for split deployment."""
        return (DistilledDepthCNNExporter(self.visual_encoder),
                DistilledFeaturesWaQExporter(self.actor, self.vae))


def load_checkpoint(actor_critic, checkpoint_file, args_file):
    model_dict = torch.load(checkpoint_file, map_location="cpu")["model_state_dict"]
    args = torch.load(args_file, map_location="cpu")["args"]
    model = actor_critic(**args)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    from rsl_rl.modules import ActorCriticDreamWaQDepth


    # This is a single distilled student, not the multi-specialist FFT policy.
    run_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs/go2_depth_waq_multiteacher_lora_distill/Sep05_12-57-33_pure_imitation",
    )
    actor_critic = load_checkpoint(
        ActorCriticDreamWaQDepth,
        os.path.join(run_dir, "model_10000.pt"),
        os.path.join(run_dir, "current_actor_args.pt"),
    )
    exporter = DistilledDepthWaQExporter(actor_critic)

    path = os.path.join(LEGGED_GYM_ROOT_DIR, "exported")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, timestamp)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"compiled_distilled_{timestamp}.pt")
    exporter.export(file)
    cnn, features = exporter.split()
    split_path = os.path.join(path, f"compiled_distilled_{timestamp}_split")
    os.makedirs(split_path, exist_ok=True)
    cnn.export(os.path.join(split_path, "DepthCNN.pt"))
    features.export(os.path.join(split_path, "FeaturesWaQ.pt"))
