import torch
import random
import numpy as np
from legged_gym.envs.base.legged_robot_dreamwaq import LeggedRobotDreamwaq
from legged_gym.utils.math_utils import wrap_to_pi, quat_apply
import torch.nn.functional as F
import torchvision.transforms as T

def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int], str) -> Tensor
    return torch.empty(*shape, device=device).uniform_(lower, upper)
    #return (upper - lower) * torch.rand() + lower

from .depth_mixin import DepthMixin

class Go2DepthWaq(DepthMixin, LeggedRobotDreamwaq):

    def _prepare_reward_function(self):
        # Keep configured zero-start curriculum terms in the reward registry.
        enabled = getattr(self.cfg.rewards, "use_reward_curriculum", False)
        if enabled:
            cfg = self.cfg.rewards.reward_curriculum
            for key in cfg.curr_reward_keys:
                if key in self.reward_scales and self.reward_scales[key] == 0:
                    low, high = cfg.curr_reward_bounds[key]
                    self.reward_scales[key] = high if high != 0 else low
        super()._prepare_reward_function()
        if enabled:
            self.step_reward_curriculum(0)

    def step_reward_curriculum(self, num_iters):
        """PACT warmup/cosine schedule, indexed by PPO iteration."""
        if not getattr(self.cfg.rewards, "use_reward_curriculum", False):
            return
        cfg = self.cfg.rewards.reward_curriculum
        if cfg.curr_steps <= 0 or cfg.warmup_steps < 0:
            raise ValueError("Reward curriculum requires curr_steps > 0 and warmup_steps >= 0")
        alpha = np.clip((num_iters - cfg.warmup_steps) / cfg.curr_steps, 0.0, 1.0)
        ramp = 0.5 * (1.0 - np.cos(np.pi * alpha))
        for key in cfg.curr_reward_keys:
            if key in self.reward_scales:
                low, high = cfg.curr_reward_bounds[key]
                value = low if alpha == 0 else high if alpha == 1 else low + (high - low) * ramp
                self.reward_scales[key] = value * self.dt
    
    def compute_observations(self):
        self._update_depth_observations()
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions ,                                                    # num_actions
        ), dim=-1)

        domain_randomization_info = torch.cat((
                    self.simulator.dr_friction_values,            # 1
                    self.simulator.dr_added_base_mass,            # 1
                    self.simulator.dr_base_com_bias,              # 3
                    self.simulator.dr_rand_push_vels[:, :2],      # 2
                    self.simulator.dr_kp_scale,                   # num_actions
                    self.simulator.dr_kd_scale,                    # num_actions
                    self.simulator.dr_motor_strength_scale,       # num_actions
            ), dim=-1)
        # Critic observation
        critic_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,                   # 3
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 43 (12-joint Go2)
        ), dim=-1)
        ## add link contact states
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                         # previous
                    self.simulator.link_contact_states,  # 17
                ),
                dim=-1,
            )
        ## add measured terrain heights
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
        # next state
        self.next_state_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions * self.cfg.control.action_scale,
        ), dim=-1)
        
        # explicit info labels
        self.explicit_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel * 0.5,  # 3
            self.simulator.link_contact_states, # contact states of hips, thighs, calves, feet and base (4+4+4+4+1)=17
            torch.clip(self.simulator.feet_pos[:, :, 2] -
                torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_height_offset, -1, 1.),  # 4
        ), dim=-1)
    

    def check_termination(self) -> None:
        """Check termination conditions and update reset buffer.
        
        Evaluates three termination conditions:
            1. Contact termination: Body contacts with termination bodies exceed threshold.
            2. Orientation termination: Projected gravity exceeds maximum allowed value.
            3. Timeout termination: Episode exceeds maximum episode length.
        
        Updates the following buffers:
            - fail_buf: Tracks consecutive failures for graceful termination.
            - time_out_buf: Indicates episodes that timed out (not actual failures).
            - reset_buf: Indicates environments needing reset.
        """
        # if the dim of link_contact_forces is 4, then it has history (IsaacLab). shape [N, T, B, 3] (N: num_envs, T: history length, B: number of links with contact sensors)
        if len(self.simulator.link_contact_forces.shape) == 4:
            self.terminated_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.termination_contact_indices, :], dim=-1), dim=1)[0]
            self.penalized_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.penalized_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_max_force_z = torch.max(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, 2], dim=1)[0]
        else:
            self.terminated_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            self.penalized_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], dim=-1)
            self.feet_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :], dim=-1)
            self.feet_max_force_z = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2]
        
        self.gap_reset_buf = self._check_unrecoverable_gap()
        fail_buf = torch.any(self.terminated_bodies_force_norm > 10.0, dim=1)
        # print(f"contact termination: {fail_buf}")
        fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity
        # print(f"gravity termination: {self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity}")
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        # print(f"time out: {self.time_out_buf}")
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.gap_reset_buf
        )


    def _check_unrecoverable_gap(self):
        if (
            not self._reset_unrecoverable_gaps
            or self.cfg.terrain.mesh_type not in ("heightfield", "trimesh")
            or not self.cfg.terrain.obtain_terrain_info_around_feet
        ):
            self.gap_fall_counter.zero_()
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        support_height = self.simulator.env_origins[:, 2].unsqueeze(1)
        terrain_under_feet = self.simulator.height_around_feet[:, :, 4]
        deep_void = terrain_under_feet < (
            support_height
            - self.cfg.termination.gap_terrain_depth_threshold
        )
        fallen_feet = deep_void & (
            self.simulator.feet_pos[:, :, 2]
            < support_height - self.cfg.termination.gap_foot_drop_threshold
        )
        enough_fallen_feet = (
            fallen_feet.sum(dim=1)
            >= self.cfg.termination.gap_min_fallen_feet
        )
        base_fallen = deep_void.any(dim=1) & (
            self.simulator.base_pos[:, 2]
            < self.simulator.env_origins[:, 2]
            - self.cfg.termination.gap_base_drop_threshold
        )
        falling_into_gap = enough_fallen_feet | base_fallen

        self.gap_fall_counter = torch.where(
            falling_into_gap,
            self.gap_fall_counter + 1,
            torch.zeros_like(self.gap_fall_counter),
        )
        return (
            self.gap_fall_counter
            >= self.cfg.termination.gap_reset_steps
        )

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_camera_envs = self.cfg.env.num_camera_envs
        if hasattr(self.cfg.sensor.depth_camera_config, "resized_resolution"): 
            self.output_resolution = self.cfg.sensor.depth_camera_config.resized_resolution
            self._has_resized_resolution = True
        else:
            H, W = self.cfg.sensor.depth_camera_config.resolution 
            t, b = self.cfg.sensor.depth_camera_config.crop_top_bottom
            l, r = self.cfg.sensor.depth_camera_config.crop_left_right
            H = H - t - b
            W = W - l - r
            self.output_resolution = (H, W)
            self._has_resized_resolution = False
        self.depth_image_size = self.output_resolution[0] * self.output_resolution[1]
        self._reset_unrecoverable_gaps = getattr(self.cfg.termination, "reset_unrecoverable_gaps", False) if hasattr(self.cfg, "termination") else False
        #depth_camera_config = self.cfg.sensor.depth_camera_config
        #self._depth_contour_thresh  = getattr(depth_camera_config, "countour_threshold", 0.)
        #self._depth_artifacts_prob  = getattr(depth_camera_config, "artifacts_prob", 0.)
        #self._depth_stereo_min      = getattr(depth_camera_config, "stereo_min_distance", 0.)
        #self._depth_sky_prob        = getattr(depth_camera_config, "sky_artifacts_prob", 0.)
        self.custom_command_curriculum = self.cfg.commands.custom_command_curriculum


    def _init_buffers(self):
        super()._init_buffers()
        #self.force_fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        #self.termination_counter_threshold = getattr(self.cfg.asset, "termination_count", 1)
        self.gap_fall_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        super()._init_depth_processing()

        # identify env ids for different terrain types
        if self.cfg.terrain.curriculum:
            terrain_types = self.simulator.terrain_types # terrains of all the envs
            terrain_type_bounds = torch.cumsum(torch.tensor(self.cfg.terrain.terrain_proportions), dim=0) * self.cfg.terrain.num_cols
            #self.slope_env_ids = ((terrain_types >=0) & (terrain_types < terrain_type_bounds[0])).nonzero(as_tuple=False).flatten()
            self.stairs_env_ids = ((terrain_types >= terrain_type_bounds[1]) & (terrain_types < terrain_type_bounds[3])).nonzero(as_tuple=False).flatten()
            self.discrete_env_ids = ((terrain_types >= terrain_type_bounds[3]) & (terrain_types < terrain_type_bounds[4])).nonzero(as_tuple=False).flatten()
            self.stepping_stones_env_ids = ((terrain_types >= terrain_type_bounds[4]) & (terrain_types < terrain_type_bounds[5])).nonzero(as_tuple=False).flatten()
            self.gaps_env_ids = ((terrain_types >= terrain_type_bounds[5]) & (terrain_types < terrain_type_bounds[6])).nonzero(as_tuple=False).flatten()
            self.pits_env_ids = ((terrain_types >= terrain_type_bounds[6]) & (terrain_types < terrain_type_bounds[7])).nonzero(as_tuple=False).flatten()
            self.high_platform_env_ids = ((terrain_types >= terrain_type_bounds[7]) & (terrain_types < terrain_type_bounds[8])).nonzero(as_tuple=False).flatten()
            self.high_platform_gaps_env_ids = ((terrain_types >= terrain_type_bounds[8]) & (terrain_types < terrain_type_bounds[9])).nonzero(as_tuple=False).flatten()
            self.center_platform_terrain_env_ids = ((terrain_types >= terrain_type_bounds[9]) & (terrain_types < terrain_type_bounds[10])).nonzero(as_tuple=False).flatten()
            # identify env ids for all heading command (others have fixed heading command specified in config)
            self.all_heading_env_ids = torch.cat((
                #self.slope_env_ids,
                self.stairs_env_ids,
                self.discrete_env_ids,
                self.gaps_env_ids,
                self.pits_env_ids, # elementary terrains with all heading commands
                self.center_platform_terrain_env_ids
            ))
            print(terrain_types, terrain_type_bounds)
            print(self.stairs_env_ids)
            print(self.gaps_env_ids)
            # identity termination base height for high_platform_gaps terrain
            difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
            self.high_platform_gaps_termination_height = eval(
                self.cfg.terrain.terrain_curriculum_difficulty["high_platform_gaps_params"]["high_platform_height"])
            
            difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
            self.platform_size = self.cfg.terrain.platform_size
            self.pit_depth_eval = self.cfg.terrain.terrain_curriculum_difficulty["pit_depth"]
            self.pit_depth = eval(self.pit_depth_eval)
        else:
            self.pit_depth_eval = None
            if self.cfg.terrain.terrain_kwargs and "depth" in self.cfg.terrain.terrain_kwargs:
                self.pit_depth = self.cfg.terrain.terrain_kwargs["depth"]

    
    def _resample_commands(self, env_ids) -> None:
        if not self._reset_unrecoverable_gaps and not self.custom_command_curriculum:
            super()._resample_commands(env_ids)
            return
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            if self.cfg.terrain.curriculum:
                # override heading command of some envs with all range heading command
                env_ids_for_heading = torch.tensor([env_id for env_id in env_ids if env_id in self.all_heading_env_ids], device=self.device)
                if len(env_ids_for_heading) > 0:
                    self.commands[env_ids_for_heading, 3] = torch_rand_float(-3.14, 3.14, (len(env_ids_for_heading), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if np.random.rand() < self.cfg.commands.zero_cmd_prob:
            self.commands[env_ids, :3] *= 0.0  # set command to zero with some probability, to encourage the robot to learn to stand still
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self._reset_unrecoverable_gaps and not self.custom_command_curriculum:
            super()._update_command_curriculum(env_ids)
            return
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if "tracking_lin_vel" in self.episode_sums and torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            # only increase upper bound of forward velocity command
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
        elif "world_vel_l2norm" in self.episode_sums and torch.mean(self.episode_sums["world_vel_l2norm"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["world_vel_l2norm"]:
                self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    
    def _pre_sim_step(self, actions):
        super()._pre_depth_step()
        return super()._pre_sim_step(actions)

    def step(self, actions):
        return *super().step(actions), self.get_depth_observations()

    def get_observations(self):
        return *super().get_observations(), self.get_depth_observations()

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        obs, privileged_obs, obs_history, explicit_labels, next_state, _, _, _, depth_sensor_output = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, obs_history, explicit_labels, next_state, depth_sensor_output

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        super()._reset_depth_buffers(env_ids)
        self.gap_fall_counter[env_ids] = 0
        
        if self.pit_depth_eval:
            difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
            self.pit_depth = eval(self.pit_depth_eval)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._resample_depth_latency()

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
        """
        
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        return noise_vec

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # refer to https://ieeexplore.ieee.org/document/11112615
        heading_error = torch.abs(self.heading - self.commands[:, 3])
        heading_coef = (1 + torch.cos(heading_error)) / 2
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.simulator.base_lin_vel[:, 0])
        # double the weight of y velocity error to discourage y axis drifting
        lin_vel_y_error = 2 * torch.square(self.commands[:, 1] - self.simulator.base_lin_vel[:, 1])
        lin_vel_error = lin_vel_x_error + lin_vel_y_error
        # add heading_coef to make the reward smaller when the robot is facing away from the commanded direction
        # thus to encourage the robot to walk across the terrain in the commanded direction
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) * heading_coef
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        
        Attention: using torch.max(self.simulator.height_around_feet) will cause reward value jumping, bad for learning
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] - torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_foot_clearance_terrain_aware(self):
        """
        Encourage swing feet to reach a terrain-aware desired height,
        while softly discouraging excessive swing height.

        Assumes:
            self.simulator.feet_pos           : (N, 4, 3)
            self.simulator.feet_vel           : (N, 4, 3)
            self.simulator._height_around_feet: (N, 4, 3, 3) or (N, 4, 9)

        Uses:
            - terrain-aware target height
            - horizontal foot velocity weighting (same style as original reward)
            - excess-height penalty to prevent over-swinging
        """

        feet_z = self.simulator.feet_pos[:, :, 2]                       # (N,4)
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)  # (N,4)

        # Flatten 3x3 terrain patch if needed, then take local max height near each foot
        h_patch = self.simulator._height_around_feet
        if h_patch.ndim == 4:   # (N,4,3,3)
            h_patch = h_patch.view(h_patch.shape[0], h_patch.shape[1], -1)  # (N,4,9)

        local_terrain_h = torch.max(h_patch, dim=-1)[0]                # (N,4)

        # Terrain-aware desired foot height
        z_des = (
            self.cfg.rewards.foot_clearance_target
            + self.cfg.rewards.foot_height_offset
            + local_terrain_h
        )                                                               # (N,4)

        # Main tracking error: encourage feet to reach desired terrain-aware height
        track_err = torch.square(feet_z - z_des)                        # (N,4)

        # Soft over-swing penalty: only penalize when foot goes too far above desired height
        # Margin gives some freedom to overshoot a little during learning
        excess_margin = 0.04  # [m], tune: 0.03 - 0.06
        excess = torch.relu(feet_z - (z_des + excess_margin))           # (N,4)
        # excess = F.softplus(feet_z - (z_des + excess_margin))           # (N,4)
        excess_err = torch.square(excess)

        # Weight excess penalty less than main tracking term
        excess_weight = 0.25  # tune: 0.1 - 0.5

        total_err = torch.sum(
            foot_vel_xy_norm * (track_err + excess_weight * excess_err),
            dim=-1
        )                                                               # (N,)

        return torch.exp(-total_err / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_base_height(self):
        # Only dedicated GAP training: inherited mixed-terrain tasks retain the
        # original reward. The standard gap's two landing surfaces are coplanar.
        if (getattr(self.cfg.rewards, "base_height_relative_to_gap_landing", False)
                and self.cfg.terrain.terrain_proportions[6] == 1.0):
            base_height = self.simulator.base_pos[:, 2] - self.simulator.env_origins[:, 2]
            return torch.square(base_height - self.cfg.rewards.base_height_target)
        return super()._reward_base_height()

    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    def _reward_feet_near_edge(self):
        """ Penalize feet being too close to the edge of a terrain
        """
        feet_near_edge = self.simulator.calc_feet_near_edge()
        feet_contact = self.feet_max_force_z > 10.0
        # print(f"feet near edge: {torch.sum(feet_near_edge)}")
        return torch.sum(feet_near_edge * feet_contact, dim=-1)

    #def _reward_base_up_pit(self):
    #    base_height = torch.clamp(self.simulator.base_pos[:, 2] - self.simulator.env_origins[:, 2], min=0.0)
    #    error = torch.square(base_height - self.cfg.rewards.base_height_target - self.pit_depth)
    #    return torch.exp(-error / self.cfg.rewards.base_up_pit_sigma)

    #def _reward_base_up_pit(self):
    #    error = torch.square(
    #        torch.clamp(self.simulator.base_pos[:, 2] - self.pit_depth, min=0.0)
    #    )
    #    sigma = self.pit_depth*self.cfg.rewards.base_up_pit_sigma
    #    return torch.exp(-error / sigma)
    
    def _reward_base_up_pit(self):
        calc = torch.mean((self.pit_depth + self.simulator.env_origins[:, 2]).unsqueeze(1) - self.simulator.measured_heights, dim=1)
        error = torch.square(calc)
        sigma = self.pit_depth * self.cfg.rewards.base_up_pit_sigma
        return torch.exp(-error / sigma)
    
    def _reward_world_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim= 1)

    def _reward_world_heading_l2norm(self):
        def wrap_to_pi(angle):
            return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi
        return torch.abs(wrap_to_pi(self.commands[:, 3] - self.heading))

    def _reward_corner_proximity(self):
        # position relative to the platform center, in the horizontal plane
        pos_rel = self.simulator.base_pos[:, :2] - self.simulator.env_origins[:, :2]
        half_size = self.cfg.terrain.platform_size / 2.0
        # coordinates of the nearest corner (sign of pos determines quadrant)
        nearest_corner = torch.sign(pos_rel) * half_size
        # true Euclidean distance from the agent to that corner
        dist_to_corner = torch.clamp(torch.norm(pos_rel - nearest_corner, dim=-1), max=half_size)
        # smooth penalty that grows as distance shrinks (tune sigma to taste)
        sigma = half_size * self.cfg.rewards.corner_proximity_sigma
        return torch.exp(-dist_to_corner / sigma)

    def _reward_alive(self):
        return 1

    def get_failure_idx(self):
        return self.reset_buf * ~self.time_out_buf
