"""Detect complete, commanded traversal of generated tire groups."""

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from .phase_observations import DogW1Phase
from .rewards import phase_gait_targets

# Shared with the mesh generator: one group contains four flat annular tires.
TIRE_OUTER_RADIUS = 0.275
TIRE_INNER_RADIUS = 0.165
TIRE_SPACING = 0.55
TIRE_HEIGHT = 0.145


def tire_approach_mask(wheel_xy, tire_centers, command_xy, heading, approach_distance,
                       wheel_radius, translation_threshold, ahead_only=True, per_tire=False):
    """Intersect a short commanded wheel path with tire annuli expanded by wheel radius.

    This is a horizontal circular approximation of the polygonal collision mesh,
    not a contact sensor. Empty holes qualify only if their rim is within reach.
    All positions are world-frame; commands are body-frame.
    """
    speed = torch.linalg.vector_norm(command_xy, dim=-1)
    cosine, sine = torch.cos(heading), torch.sin(heading)
    velocity = torch.stack((cosine * command_xy[:, 0] - sine * command_xy[:, 1],
                            sine * command_xy[:, 0] + cosine * command_xy[:, 1]), dim=-1)
    direction = velocity / speed.clamp_min(1.e-6)[:, None]
    relative = wheel_xy[:, :, None, :] - tire_centers[:, None, :, :]
    along = (relative * direction[:, None, None, :]).sum(dim=-1)
    closest_t = (-along).clamp(0.0, approach_distance)
    closest = relative + closest_t[..., None] * direction[:, None, None, :]
    endpoint = relative + approach_distance * direction[:, None, None, :]
    minimum = torch.linalg.vector_norm(closest, dim=-1)
    maximum = torch.maximum(torch.linalg.vector_norm(relative, dim=-1),
                            torch.linalg.vector_norm(endpoint, dim=-1))
    hit = (minimum <= TIRE_OUTER_RADIUS + wheel_radius)
    hit &= maximum >= max(0.0, TIRE_INNER_RADIUS - wheel_radius)
    if ahead_only:
        hit &= along <= 0.0  # Stop rewarding a tire once its center is behind this wheel.
    hit &= (speed > translation_threshold)[:, None, None]
    return hit if per_tire else (hit.sum(dim=-1) > 0)


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


class TireObstacleClimbReward(ManagerTermBase):
    """One bounded response per wheel/tire/episode, sharing the observation clock."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        bodies = cfg.params["asset_cfg"]
        joints = cfg.params["joint_cfg"]
        sensor = cfg.params["sensor_cfg"]
        names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        if (not bodies.preserve_order or not joints.preserve_order or not sensor.preserve_order
                or list(bodies.body_names) != names or list(sensor.body_names) != names
                or list(joints.joint_names) != [name + "_joint" for name in names]):
            raise ValueError("TireObstacleClimbReward requires matching ordered wheel bodies, joints and contacts.")
        p = cfg.params
        for name in ("response_duration", "reward_budget", "speed_reference", "lift_std",
                     "progress_reference", "slip_std", "wheel_radius"):
            if p.get(name, 1.0) <= 0:
                raise ValueError(f"TireObstacleClimbReward {name} must be positive.")
        if p.get("approach_distance", 0.12) < 0 or p.get("lift_clearance", 0.03) < 0:
            raise ValueError("Tire approach distance and clearance must be nonnegative.")
        if p.get("speed_weight", 0.6) < 0 or p.get("lift_weight", 2.0) < 0:
            raise ValueError("Bounded tire guidance requires nonnegative component weights.")
        self.env = env
        self.centers = torch.as_tensor(env.scene.terrain.cfg.terrain_generator.group_centers,
                                       dtype=torch.float, device=env.device)
        self.offsets = self.centers.new_tensor([
            [-TIRE_SPACING / 2, -TIRE_SPACING / 2], [TIRE_SPACING / 2, -TIRE_SPACING / 2],
            [-TIRE_SPACING / 2, TIRE_SPACING / 2], [TIRE_SPACING / 2, TIRE_SPACING / 2],
        ])
        shape = (env.num_envs, 4)
        self.triggered = torch.zeros(*shape, self.centers.shape[2] * 4, dtype=torch.bool, device=env.device)
        self.active = torch.full(shape, -1, dtype=torch.long, device=env.device)
        self.time_left = torch.zeros(*shape, device=env.device)
        self.earned = torch.zeros(*shape, device=env.device)
        self.direction = torch.zeros(*shape, 2, device=env.device)
        self.previous = torch.zeros(env.num_envs, 2, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.env_ids = torch.arange(env.num_envs, device=env.device)[:, None]
        self.wheel_ids = torch.arange(4, device=env.device)[None, :]
        self.clock = DogW1Phase(cfg, env)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.triggered[ids] = False
        self.active[ids] = -1
        self.time_left[ids] = 0.0
        self.earned[ids] = 0.0
        self.direction[ids] = 0.0
        self.initialized[ids] = False
        self.clock.reset(env_ids)

    def __call__(self, env, asset_cfg, joint_cfg, sensor_cfg, command_name,
                 cycle_time, stop_cycle_time, stop_duration, translation_threshold, yaw_threshold,
                 stance_ratio, phase_offset=0.5, approach_distance=0.12, ahead_only=True,
                 response_duration=0.8, reward_budget=0.4, speed_reference=22.0,
                 wheel_radius=0.0765, speed_weight=0.6, lift_weight=2.0,
                 lift_clearance=0.03, lift_std=0.05, progress_threshold=0.02,
                 progress_reference=0.2, contact_threshold=1.0, slip_std=0.3):
        self.clock(env, command_name, cycle_time, stop_cycle_time, stop_duration,
                   translation_threshold, yaw_threshold)
        terrain = env.scene.terrain
        centers = self.centers[terrain.terrain_levels, terrain.terrain_types] + terrain.env_origins[:, None, :2]
        tires = (centers[:, :, None, :] + self.offsets[None, None, :, :]).flatten(1, 2)
        data = env.scene[asset_cfg.name].data
        positions = data.body_pos_w[:, asset_cfg.body_ids]
        command = env.command_manager.get_command(command_name)
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        c, s = torch.cos(data.heading_w), torch.sin(data.heading_w)
        velocity = torch.stack((c * command[:, 0] - s * command[:, 1],
                                s * command[:, 0] + c * command[:, 1]), dim=-1)
        direction = velocity / speed.clamp_min(1.e-6)[:, None]
        hits = tire_approach_mask(positions[..., :2], tires, command[:, :2], data.heading_w,
                                 approach_distance, wheel_radius, translation_threshold, ahead_only,
                                 per_tire=True)
        candidates = hits & ~self.triggered
        distances = torch.linalg.vector_norm(positions[:, :, None, :2] - tires[:, None, :, :], dim=-1)
        selected = torch.where(candidates, distances, torch.full_like(distances, float("inf"))).argmin(dim=-1)
        start = (self.active < 0) & candidates.any(dim=-1)
        self.active = torch.where(start, selected, self.active)
        self.triggered[self.env_ids, self.wheel_ids, selected] |= start
        self.time_left = torch.where(start, response_duration, self.time_left)
        self.earned = torch.where(start, 0.0, self.earned)
        self.direction = torch.where(start[..., None], direction[:, None, :], self.direction)

        center = tires[self.env_ids, self.active.clamp_min(0)]
        relative = positions[..., :2] - center
        along = (relative * self.direction).sum(dim=-1)
        lateral = (relative[..., 0] * self.direction[..., 1] - relative[..., 1] * self.direction[..., 0]).abs()
        alignment = (self.direction * direction[:, None, :]).sum(dim=-1)
        active = ((self.active >= 0) & (self.time_left > 0) & (speed > translation_threshold)[:, None]
                  & (alignment > 0.5) & (along < TIRE_OUTER_RADIUS + wheel_radius)
                  & (along > -TIRE_OUTER_RADIUS - wheel_radius - 2 * approach_distance)
                  & (lateral < TIRE_OUTER_RADIUS + wheel_radius))

        # Require both measured base motion and displacement: leg extension alone is not progress.
        displacement = data.root_pos_w[:, :2] - self.previous
        measured = (data.root_lin_vel_w[:, None, :2] * self.direction).sum(dim=-1)
        delta_speed = (displacement[:, None, :] * self.direction).sum(dim=-1) / env.step_dt
        progress = torch.minimum(measured, delta_speed).clamp_min(0.0)
        progress_ok = (progress > progress_threshold) & self.initialized[:, None]
        progress_score = (progress / progress_reference).clamp(0.0, 1.0) * progress_ok
        self.previous.copy_(data.root_pos_w[:, :2])
        self.initialized[:] = True

        # Current URDF: all wheel joint axes are local +Z. Rotate each axle into world frame.
        local_axle = torch.zeros_like(positions)
        local_axle[..., 2] = 1.0
        axle = quat_apply(data.body_quat_w[:, asset_cfg.body_ids], local_axle)
        omega = data.joint_vel[:, joint_cfg.joint_ids]
        rolling = wheel_radius * omega[..., None] * torch.stack((axle[..., 1], -axle[..., 0]), dim=-1)
        rolling_forward = (rolling * self.direction).sum(dim=-1)
        wheel_forward = (data.body_lin_vel_w[:, asset_cfg.body_ids, :2] * self.direction).sum(dim=-1)
        contact_data = env.scene.sensors[sensor_cfg.name].data
        contact = ((contact_data.current_contact_time[:, sensor_cfg.body_ids] > 0.0)
                   & (torch.linalg.vector_norm(contact_data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > contact_threshold))
        speed_score = torch.exp(-((rolling_forward - wheel_radius * speed_reference)
                                  / (wheel_radius * speed_reference)).square())
        slip_score = torch.exp(-((rolling_forward - wheel_forward) / slip_std).square())
        drive = speed_score * slip_score * contact * (rolling_forward > 0.0) * progress_score

        # Target the actual tire top; excessive height loses reward. Only the shared swing phase earns lift.
        stance, envelope = phase_gait_targets(self.clock.phase, stance_ratio, 1.0, phase_offset, 0.0)
        bottom = positions[..., 2] - env.scene.env_origins[:, 2:3] - wheel_radius
        target = TIRE_HEIGHT + lift_clearance
        lift_score = torch.exp(-((bottom - target) / lift_std).square())
        lift = lift_score * (~stance) * envelope
        raw = (speed_weight * drive + lift_weight * lift) * active
        # Budget is integrated per wheel before the mean and outer RewardTerm weight.
        duration = self.time_left.clamp(0.0, env.step_dt)
        increment = torch.minimum(raw * duration, (reward_budget - self.earned).clamp_min(0.0))
        self.earned += increment
        self.time_left = (self.time_left - env.step_dt).clamp_min(0.0)
        finished = ~active | (self.time_left <= 0) | (self.earned >= reward_budget)
        self.active = torch.where(finished, -1, self.active)
        return increment.mean(dim=1) / env.step_dt


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
