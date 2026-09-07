import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.exp_data_logger import ExpLogger

import numpy as np
import torch
from rsl_rl.utils.LoRA import SequentialMultiLora, MultiLora
import copy

import argparse

import cv2

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

import copy

class DepthCNNExporterWaQ(torch.nn.Module):
    """Standalone exportable module for just the depth visual encoders."""

    @torch.jit.unused
    def __init__(self, visual_encoders):
        super().__init__()
        self.visual_encoders = visual_encoders

        self.visual_encoder = self.visual_encoders[0]

    @torch.jit.export
    def swap(self, index: int):
        index = index + 1
        for i, ve in enumerate(self.visual_encoders):
            if i == index:
                self.visual_encoder = ve

    @torch.jit.export
    def forward(self, depth_image):
        return self.visual_encoder(depth_image)

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filename)


class PolicyExporterFeaturesWaQ(torch.nn.Module):
    """Standalone exportable module for actor + vae, taking a precomputed visual_latent."""

    @torch.jit.unused
    def __init__(self, actors, vaes):
        super().__init__()
        self.actors = actors
        self.vaes = vaes

        self.actor = self.actors[0]
        self.vae = self.vaes[0]

    @torch.jit.export
    def swap(self, index: int):
        index = index + 1
        for i, (a, v) in enumerate(zip(self.actors, self.vaes)):
            if i == index:
                self.actor = a
                self.vae = v

    @torch.jit.export
    def forward(self, observations, obs_history, visual_latent):
        mean_out = self.vae.inference(obs_history)
        return self.actor(
            torch.cat(
                (observations, mean_out, visual_latent),
                dim=-1,
            )
        )

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filename)


class PolicyExporterDepthWaQ(torch.nn.Module):
    @torch.jit.unused
    def __init__(self, actor_critics):
        super().__init__()
        self.actors = torch.nn.ModuleList([
            copy.deepcopy(ac.actor)
            for ac in actor_critics
        ])

        self.vaes = torch.nn.ModuleList([
            copy.deepcopy(ac.vae)
            for ac in actor_critics
        ])

        self.visual_encoders = torch.nn.ModuleList([
            copy.deepcopy(ac.visual_encoder)
            for ac in actor_critics
        ])

        self.actor = self.actors[0]
        self.vae = self.vaes[0]
        self.visual_encoder = self.visual_encoders[0]

        self.num_of_loras = len(self.actors)

    @torch.jit.export
    def swap(self, index: int):
        index = index + 1
        for i, (a, v, ve) in enumerate(zip(self.actors, self.vaes, self.visual_encoders)):
            if i == index:
                self.actor = a
                self.vae = v
                self.visual_encoder = ve

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
    def split_cnn(self):
        """
        Split the already-constructed policy into a standalone CNN exporter
        and a standalone actor+vae exporter. No new modules are constructed,
        both share references to the same underlying ModuleLists.
        """
        cnn = DepthCNNExporterWaQ(self.visual_encoders)
        main = PolicyExporterFeaturesWaQ(self.actors, self.vaes)
        return cnn, main


import os
import copy
import torch    

def loader(actor_critic, file):
    model_dict, args = torch.load(file[0])["model_state_dict"], torch.load(file[1])["args"]
    print(args)
    model = actor_critic(**args)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    from rsl_rl.modules import ActorCriticDreamWaQDepth


    models = [
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_baseline/Aug09_02-33-11_dreamwaq_isaacgym", 7000),
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_fft_gap/Aug12_17-01-51_dreamwaq_isaacgym", 47000),
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_fft_all_stairs/Aug14_14-23-53_dreamwaq_isaacgym", 47000),
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_fft_pit/Aug29_00-30-31_dreamwaq_isaacgym", 67000)
    ]

    model_files = [
        (os.path.join(folder, f"model_{model}.pt"), os.path.join(folder, f"current_actor_args.pt"))
        for folder, model in models

    ]

    actor_critics = [
        loader(ActorCriticDreamWaQDepth, file)
        for file in model_files
    ]

    exporter = PolicyExporterDepthWaQ(actor_critics)

    #print(exporter)
    #observations=torch.randn(1, 45)
    #obs_history=torch.randn(1, 900)
    #depth_image=torch.randn(1, 1, 48, 64)
    #output = exporter(observations, obs_history, depth_image)


    #test_swap_eager_and_exported(exporter, tmp_export_dir="/tmp/lora_export_test")

    path = os.path.join(LEGGED_GYM_ROOT_DIR, "exported")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"{timestamp}_multi_fft")
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"compiled_multi_fft_{timestamp}.pt")
    exporter.export(file)
    cnn, main = exporter.split_cnn()
    path = os.path.join(path, f"compiled_multi_fft_{timestamp}_split")
    os.makedirs(path, exist_ok=True)
    cnn.export(os.path.join(path, f"DepthCNN.pt"))
    main.export(os.path.join(path, f"FeaturesWaQ.pt"))
    
    #policy = torch.jit.load(os.path.join(path, f"compiled_lora_{timestamp}.pt"))
    #print(
    #    torch.sum(output - policy(observations, obs_history, depth_image))
    #)
            