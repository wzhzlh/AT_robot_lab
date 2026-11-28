"""Filtered-contact-triggered wall response; tunable values live in wall_env_cfg."""

import torch
from isaaclab.managers import ManagerTermBase
from .phase_observations import DogW1Phase


def response_transition(stage, elapsed, force, crossed, cleared, triggered, dt, threshold, timeout):
    """Stages 0 approach, 1 front response, 2 hind response, 3 recovery; 4 failure."""
    entering = (stage == 0) & ~triggered & (force > threshold)
    stage = torch.where(entering, 1, stage)
    triggered = triggered | entering
    active = (stage == 1) | (stage == 2)
    elapsed = torch.where(active, elapsed + dt, elapsed)
    stage = torch.where((stage == 1) & crossed[:, :2].all(dim=1), 2, stage)
    stage = torch.where(((stage == 1) | (stage == 2)) & cleared, 3, stage)
    stage = torch.where(((stage == 1) | (stage == 2)) & (elapsed >= timeout), 4, stage)
    return stage, elapsed, triggered


def filtered_force(sensor):
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        raise RuntimeError("Wall response requires contact sensors filtered to Wall, not net ground forces.")
    return matrix.sum(dim=(1, 2))


def contact_tangent_velocity(com_velocity, angular_velocity, center, com, normal, radius):
    offset = center - com - radius * normal
    velocity = com_velocity + torch.cross(angular_velocity, offset, dim=-1)
    return velocity - (velocity * normal).sum(dim=-1, keepdim=True) * normal


class WallContactState(ManagerTermBase):
    """Own state once per control step; expose separate named reward components."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.params = cfg.params
        self.clock = DogW1Phase(cfg, env)
        self.stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.triggered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.crossed = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self.best_base = torch.zeros(env.num_envs, device=env.device)
        self.best_feet = torch.zeros(env.num_envs, 4, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.base_contact_time = torch.zeros(env.num_envs, device=env.device)
        self.last_step = -1
        self.values = {}

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for tensor in (self.stage, self.elapsed, self.triggered, self.crossed, self.best_base,
                       self.best_feet, self.initialized, self.base_contact_time):
            tensor[ids] = 0
        for value in self.values.values():
            value[ids] = 0
        self.clock.reset(env_ids)

    def evaluate(self, env):
        if self.last_step == env.common_step_counter:
            return
        self.last_step = env.common_step_counter
        p = self.params
        robot = env.scene["robot"]
        data = robot.data
        feet_ids = p["feet_cfg"].body_ids
        base = data.root_pos_w - env.scene.env_origins
        feet = data.body_pos_w[:, feet_ids] - env.scene.env_origins[:, None, :]
        self.best_base = torch.where(self.initialized, self.best_base, base[:, 0])
        self.best_feet = torch.where(self.initialized[:, None], self.best_feet, feet[..., 0])
        self.initialized[:] = True
        dx = (base[:, 0] - self.best_base).clamp_min(0.)
        foot_dx = (feet[..., 0] - self.best_feet).clamp_min(0.)
        self.best_base = torch.maximum(self.best_base, base[:, 0])
        self.best_feet = torch.maximum(self.best_feet, feet[..., 0])
        wall_forces = torch.stack([filtered_force(env.scene.sensors[name]) for name in p["wheel_sensors"]], dim=1)
        trigger_force = torch.stack([
            torch.linalg.vector_norm(filtered_force(env.scene.sensors[name]), dim=-1)
            for name in p["trigger_sensors"]
        ], dim=1).sum(dim=1)
        base_force = torch.linalg.vector_norm(filtered_force(env.scene.sensors[p["base_sensor"]]), dim=-1)
        far = p["wall_x"] + p["wall_thickness"] / 2
        above = feet[..., 2] - p["foot_radius"] > p["wall_height"] + p["clearance_margin"]
        crossing = (feet[..., 0] > far) & above & (feet[..., 1].abs() < p["wall_length"] / 2)
        new_crossing = crossing & ~self.crossed
        # Crossing before a filtered wall contact cannot bootstrap the response state.
        eligible = self.triggered | (trigger_force > p["contact_threshold"])
        self.crossed |= crossing & eligible[:, None]
        cleared = (base[:, 0] > far + p["base_margin"]) & (feet[..., 0] > far + p["foot_margin"]).all(dim=1)
        lane = base[:, 1].abs() < p["lane_half_width"]
        self.stage, self.elapsed, self.triggered = response_transition(
            self.stage, self.elapsed, trigger_force, self.crossed, cleared & lane,
            self.triggered, env.step_dt, p["contact_threshold"], p["response_timeout"],
        )
        active = ((self.stage == 1) | (self.stage == 2)) & lane
        self.base_contact_time = torch.where(
            active & (base_force > p["contact_threshold"]), self.base_contact_time + env.step_dt, 0.,
        )
        self.stage = torch.where(self.base_contact_time > p["max_base_contact_time"], 4, self.stage)
        active &= self.stage != 4
        front = torch.tensor([1., 1., 0., 0.], device=env.device)
        hind = 1. - front
        selected = torch.where((self.stage == 1)[:, None], front[None, :], hind[None, :]) * active[:, None]
        near = (feet[..., 0] - p["wall_x"]).abs() < p["wall_thickness"] / 2 + p["near_margin"]
        forward = (dx / (env.step_dt * p["target_speed"])).clamp(0., 1.)
        foot_forward = (foot_dx / (env.step_dt * p["target_speed"])).clamp(0., 1.)
        phase_params = {key: p[key] for key in ("command_name", "cycle_time", "stop_cycle_time",
                       "stop_duration", "translation_threshold", "yaw_threshold")}
        self.clock(env, **phase_params)
        phase = self.clock.phase
        opposite = (phase + p["phase_offset"]) % 1.
        leg_phase = torch.stack((phase, opposite, opposite, phase), dim=1)
        swing = leg_phase >= p["stance_ratio"]
        contact = env.scene.sensors[p["contact_cfg"].name].data.current_contact_time[:, p["contact_cfg"].body_ids] > 0.
        hip = data.joint_pos[:, p["hip_cfg"].joint_ids] - data.default_joint_pos[:, p["hip_cfg"].joint_ids]
        signs = torch.tensor(p["hip_outward_signs"], device=env.device)
        hip_score = torch.exp(-.5 * ((hip * signs - p["hip_target"]) / p["hip_sigma"]).square())
        lift_error = feet[..., 2] - (p["wall_height"] + p["foot_radius"] + p["clearance_margin"])
        lift_score = torch.exp(-.5 * (lift_error / p["lift_sigma"]).square())
        lift_gate = selected * near * swing * ~contact
        self.values["hip"] = (hip_score * selected * near * forward[:, None]).sum(dim=1) / 2
        self.values["lift"] = (lift_score * lift_gate * torch.maximum(foot_forward, forward[:, None])).sum(dim=1) / 2
        self.values["clear"] = (new_crossing * eligible[:, None] * lane[:, None]).float().sum(dim=1) / env.step_dt
        # Signed rolling speed in world +X is omega_y * radius; no hardcoded joint signs.
        omega = data.body_ang_vel_w[:, feet_ids]
        rolling = omega[..., 1] * p["foot_radius"]
        drive_score = torch.exp(-.5 * ((rolling - p["target_speed"]) / p["wheel_speed_sigma"]).square())
        self.values["wheel"] = (drive_score * contact * near).mean(dim=1) * active * forward
        self.values["progress"] = dx * active / env.step_dt
        normal = torch.zeros_like(wall_forces)
        normal[..., 2] = 1.
        magnitude = torch.linalg.vector_norm(wall_forces, dim=-1, keepdim=True)
        normal = torch.where(magnitude > p["contact_threshold"], wall_forces / magnitude.clamp_min(1e-6), normal)
        tangent = contact_tangent_velocity(
            data.body_lin_vel_w[:, feet_ids], omega, data.body_pos_w[:, feet_ids],
            data.body_com_pos_w[:, feet_ids], normal, p["foot_radius"],
        )
        self.values["slip"] = (torch.linalg.vector_norm(tangent, dim=-1) * contact).mean(dim=1) * active
        self.values["collision"] = (base_force - p["contact_threshold"]).clamp_min(0.).clamp(max=p["collision_force_cap"]) * active
        recovery = (self.stage == 3) & lane
        upright = (-data.projected_gravity_b[:, 2]).clamp(0., 1.)
        self.values["recovery"] = recovery * upright * torch.exp(-data.root_ang_vel_b.square().sum(dim=1))

    def __call__(self, env, feet_cfg, hip_cfg, contact_cfg, wheel_sensors, trigger_sensors, base_sensor,
                 wall_x, wall_height, wall_thickness, wall_length, foot_radius, clearance_margin,
                 contact_threshold, response_timeout, max_base_contact_time, base_margin, foot_margin,
                 lane_half_width, near_margin, target_speed, hip_outward_signs, hip_target, hip_sigma,
                 lift_sigma, wheel_speed_sigma, collision_force_cap, command_name, cycle_time,
                 stop_cycle_time, stop_duration, translation_threshold, yaw_threshold, phase_offset, stance_ratio):
        self.evaluate(env)
        return self.values["hip"]


def wall_motion_component(env, component):
    state = env.reward_manager.get_term_cfg("wall_response_hip").func
    state.evaluate(env)
    return state.values[component]


def wall_response_failed(env):
    state = env.reward_manager.get_term_cfg("wall_response_hip").func
    state.evaluate(env)
    return state.stage == 4
