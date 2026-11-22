"""Detect complete, commanded traversal of generated tire groups."""

import torch
from isaaclab.managers import ManagerTermBase


class TireTraversalTracker:
    def __init__(self, num_envs, num_groups, device):
        self.active = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.axis = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.sign = torch.ones(num_envs, device=device)
        self.previous = torch.zeros(num_envs, 2, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.visited = torch.zeros(num_envs, num_groups, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        self.active[env_ids] = -1
        self.initialized[env_ids] = False
        self.visited[env_ids] = False

    def update(self, position, velocity_command, centers, half_extent, min_speed, failed):
        count = position.shape[0]
        ids = torch.arange(count, device=position.device)
        moving = torch.linalg.vector_norm(velocity_command, dim=1) >= min_speed
        relative = position[:, None, :] - centers
        prior = self.previous[:, None, :] - centers
        axis = velocity_command.abs().argmax(dim=1)
        sign = torch.sign(velocity_command.gather(1, axis[:, None]).squeeze(1))
        forward = relative.gather(2, axis[:, None, None].expand(-1, centers.shape[1], 1)).squeeze(2) * sign[:, None]
        old_forward = prior.gather(2, axis[:, None, None].expand(-1, centers.shape[1], 1)).squeeze(2) * sign[:, None]
        lateral = relative.gather(2, (1 - axis)[:, None, None].expand(-1, centers.shape[1], 1)).squeeze(2)
        old_lateral = prior.gather(2, (1 - axis)[:, None, None].expand(-1, centers.shape[1], 1)).squeeze(2)
        entry = (old_forward < -half_extent) & (forward >= -half_extent) & (forward < half_extent)
        entry &= (lateral.abs() < half_extent) & (old_lateral.abs() < half_extent) & ~self.visited
        entry &= (moving & self.initialized & ~failed & (self.active < 0))[:, None]
        has_entry = entry.any(dim=1)
        chosen = entry.float().argmax(dim=1)
        self.active = torch.where(has_entry, chosen, self.active)
        self.axis = torch.where(has_entry, axis, self.axis)
        self.sign = torch.where(has_entry, sign, self.sign)
        group = self.active.clamp_min(0)
        current = relative[ids, group]
        previous = prior[ids, group]
        along = current.gather(1, self.axis[:, None]).squeeze(1) * self.sign
        before = previous.gather(1, self.axis[:, None]).squeeze(1) * self.sign
        side = current.gather(1, (1 - self.axis)[:, None]).squeeze(1)
        command_forward = velocity_command.gather(1, self.axis[:, None]).squeeze(1) * self.sign
        active = self.active >= 0
        valid = moving & (command_forward >= min_speed) & ~failed & (side.abs() < half_extent)
        success = active & valid & (before < half_extent) & (along >= half_extent)
        self.visited[ids, group] |= success
        cancel = active & (~valid | (along < -half_extent))
        self.active = torch.where(success | cancel, -1, self.active)
        self.previous.copy_(position)
        self.initialized[:] = True
        return success


class TireTraversalReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        centers = terrain.cfg.terrain_generator.group_centers
        self.centers = torch.as_tensor(centers, dtype=torch.float, device=env.device)
        self.tracker = TireTraversalTracker(env.num_envs, self.centers.shape[2], env.device)

    def reset(self, env_ids=None):
        self.tracker.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env, command_name, half_extent, min_speed):
        terrain = env.scene.terrain
        centers = self.centers[terrain.terrain_levels, terrain.terrain_types] + terrain.env_origins[:, None, :2]
        robot = env.scene["robot"]
        heading = robot.data.heading_w
        command = env.command_manager.get_command(command_name)
        cosine, sine = torch.cos(heading), torch.sin(heading)
        velocity = torch.stack((cosine * command[:, 0] - sine * command[:, 1],
                                sine * command[:, 0] + cosine * command[:, 1]), dim=1)
        success = self.tracker.update(robot.data.root_pos_w[:, :2], velocity, centers, half_extent,
                                      min_speed, env.termination_manager.terminated)
        return success.float() / env.step_dt
