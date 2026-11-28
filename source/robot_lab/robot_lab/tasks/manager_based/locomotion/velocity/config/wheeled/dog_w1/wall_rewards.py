"""Fixed-wall traversal objectives in environment-local world coordinates."""

import torch
from isaaclab.managers import ManagerTermBase


def crossing_conditions(base, feet, gravity_z, angular_speed, wall_x, thickness, half_length, margin,
                        foot_margin, lane_margin, upright_threshold, max_angular_speed,
                        min_base_height, max_base_height, max_foot_height):
    """Require rear chassis clearance, every wheel behind wall, and upright landing."""
    far_face = wall_x + thickness / 2
    crossed = (base[:, 0] > far_face + margin) & (feet[..., 0] > far_face + foot_margin).all(dim=1)
    in_lane = (base[:, 1].abs() < half_length - lane_margin) & (feet[..., 1].abs() < half_length).all(dim=1)
    stable = (gravity_z < -upright_threshold) & (angular_speed < max_angular_speed) & (base[:, 2] > min_base_height) & (base[:, 2] < max_base_height)
    landed = (feet[..., 2] < max_foot_height).all(dim=1)
    return crossed & in_lane & stable & landed


class WallTraversalReward(ManagerTermBase):
    """Monotonic progress plus one success bonus; no reward for oscillating at wall."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.params = cfg.params
        self.last_step = -1
        self.best_x = torch.zeros(env.num_envs, device=env.device)
        self.stable_time = torch.zeros_like(self.best_x)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.bonus = torch.zeros_like(self.best_x)
        self.progress = torch.zeros_like(self.best_x)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best_x[ids] = 0.
        self.stable_time[ids] = 0.
        self.success[ids] = False
        self.bonus[ids] = 0.
        self.progress[ids] = 0.
        # Keep last_step: other environments must not advance twice on partial reset.

    def evaluate(self, env):
        if self.last_step == env.common_step_counter:
            return
        self.last_step = env.common_step_counter
        p = self.params
        data = env.scene["robot"].data
        base = data.root_pos_w - env.scene.env_origins
        feet = data.body_pos_w[:, p["asset_cfg"].body_ids] - env.scene.env_origins[:, None, :]
        lane = base[:, 1].abs() < p["wall_length"] / 2 - p["lane_margin"]
        x = base[:, 0].clamp(0., p["wall_x"] + p["progress_limit"])
        self.progress = (x - self.best_x).clamp_min(0.) * lane
        self.best_x = torch.maximum(self.best_x, x)
        candidate = crossing_conditions(
            base, feet, data.projected_gravity_b[:, 2],
            torch.linalg.vector_norm(data.root_ang_vel_b, dim=1),
            p["wall_x"], p["wall_thickness"], p["wall_length"] / 2, p["base_margin"],
            p["foot_margin"], p["lane_margin"], p["upright_threshold"], p["max_angular_speed"],
            p["min_base_height"], p["max_base_height"], p["max_foot_height"],
        )
        candidate &= torch.linalg.vector_norm(data.root_lin_vel_b, dim=1) < p["max_linear_speed"]
        self.stable_time = torch.where(candidate, self.stable_time + env.step_dt, 0.)
        success = self.stable_time >= p["stable_duration"]
        self.bonus = (success & ~self.success).float()
        self.success |= success

    def __call__(self, env, asset_cfg, wall_x, wall_thickness, wall_length,
                 progress_scale, success_bonus, progress_limit, base_margin, foot_margin,
                 lane_margin, upright_threshold, max_angular_speed, max_linear_speed,
                 min_base_height, max_base_height, max_foot_height, stable_duration):
        self.evaluate(env)
        # Isaac Lab multiplies reward terms by dt; preserve discrete progress/bonus.
        return (progress_scale * self.progress + success_bonus * self.bonus) / env.step_dt


def wall_success(env):
    term = env.reward_manager.get_term_cfg("wall_task").func
    term.evaluate(env)
    return term.success


def wall_out_of_lane(env, half_width, min_x, max_x):
    base = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return (base[:, 1].abs() > half_width) | (base[:, 0] > max_x) | (base[:, 0] < min_x)


def wall_clearance_error(env, asset_cfg, wall_x, wall_thickness, wall_height, foot_radius, near_margin, clearance_margin):
    """Only wheels near the wall need to clear its top; no unbounded height reward."""
    data = env.scene["robot"].data
    feet = data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
    near = (feet[..., 0] - wall_x).abs() < wall_thickness / 2 + near_margin
    deficit = (wall_height + foot_radius + clearance_margin - feet[..., 2]).clamp_min(0.)
    return (deficit.square() * near).mean(dim=1)


def wall_base_clearance(env, wall_x, wall_height, overlap_half_width, clearance_margin):
    base = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    overlap = (base[:, 0] - wall_x).abs() < overlap_half_width
    return (wall_height + clearance_margin - base[:, 2]).clamp_min(0.).square() * overlap
