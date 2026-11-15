"""Dog2 stair rewards; geometry/velocities/forces are all in world coordinates.

No policy observations are added. Five manager terms share one query and swing
state. Length errors are normalized before weighting; all terms are penalties.
"""

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCaster
from robot_lab.terrain_raycast import raycast_mesh

from robot_lab.gait import foot_geometry
from .periodic_gait import FOOT_NAMES, FOOT_OFFSETS


class _StairGeometry:
    def __init__(self, env, asset_cfg, sensor_cfg, terrain_sensor_cfg, command_name, foot_offsets):
        self.env = env
        self.asset = env.scene[asset_cfg.name]
        self.ids = asset_cfg.body_ids
        self.sensor = env.scene.sensors[sensor_cfg.name]
        self.contact_ids = sensor_cfg.body_ids
        self.scanner = env.scene.sensors[terrain_sensor_cfg.name]
        self.command_name = command_name
        self.offsets = torch.tensor(foot_offsets, device=env.device)
        if len(self.ids) != 4 or len(self.contact_ids) != 4 or self.offsets.shape != (4, 3):
            raise ValueError("Dog2 rewards require ordered FL, FR, RL, RR foot bodies and contacts")
        self.footprint_offsets = self.offsets.new_tensor(((0, 0), (.04, 0), (-.04, 0), (0, .04), (0, -.04)))
        self.sample_index = torch.arange(12, device=env.device)
        self.mesh = None
        self.distances = torch.linspace(-0.06, 0.30, 13, device=env.device)
        # Scratch rays are overwritten in full on every update. Keep their
        # storage and constant direction instead of allocating clones + cat.
        self.ray_starts = torch.empty((env.num_envs, 4, 18, 3), device=env.device)
        self.ray_directions = torch.zeros_like(self.ray_starts)
        self.ray_directions[..., 2] = -1
        shape = (env.num_envs, 4)
        self.was_swing = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.locked = torch.zeros_like(self.was_swing)
        self.edge = torch.zeros((*shape, 2), device=env.device)
        self.direction = torch.zeros_like(self.edge)
        self.launch = torch.zeros(shape, device=env.device)
        self.top = torch.zeros_like(self.launch)
        self.target = torch.zeros_like(self.edge)
        self.reference = torch.zeros(env.num_envs, device=env.device)
        self.reference_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.step = None

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.was_swing[ids] = False
        self.locked[ids] = False
        self.reference_valid[ids] = False
        self.step = None

    def update(self):
        env = self.env
        if self.step == env.common_step_counter:
            return
        self.step = env.common_step_counter
        data = self.asset.data
        self.clock = env.command_manager.get_term(self.command_name).gait_clock
        self.stance, self.swing, self.progress = self.clock.foot_targets()
        self.active = self.clock.active
        self.center, extent = foot_geometry(
            data.body_link_pos_w[:, self.ids], data.body_link_quat_w[:, self.ids],
            self.offsets, 0.0384, 0.014,
        )
        self.sole = self.center[..., 2] - extent
        lever = self.center - data.body_link_pos_w[:, self.ids]
        self.velocity = data.body_link_lin_vel_w[:, self.ids] + torch.cross(
            data.body_link_ang_vel_w[:, self.ids], lever, dim=-1,
        )
        self.force = self.sensor.data.net_forces_w[:, self.contact_ids]
        self.contact = self.sensor.data.current_contact_time[:, self.contact_ids] > 0
        self.touchdown = self.sensor.compute_first_contact(env.step_dt)[:, self.contact_ids]
        command = env.command_manager.get_command(self.command_name)[:, :2]
        yaw = data.heading_w
        direction = torch.stack((yaw.cos() * command[:, 0] - yaw.sin() * command[:, 1],
                                 yaw.sin() * command[:, 0] + yaw.cos() * command[:, 1]), dim=-1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        # 13 longitudinal samples + a five-point sole footprint. The sampling
        # line follows travel, so backwards and lateral commands work as well.
        starts = self.ray_starts
        starts[..., :13, :2] = self.center[:, :, None, :2] + direction[:, None, None, :] * self.distances[None, None, :, None]
        starts[..., 13:, :2] = self.center[:, :, None, :2] + self.footprint_offsets
        # Cast from above both the feet and base, not from inside the riser.
        starts[..., 2] = torch.maximum(self.center[..., 2], data.root_pos_w[:, None, 2])[:, :, None] + .5
        # Mesh registration happens when the sensor initializes, independently
        # of data updates. Do not trigger the unused 187-ray critic grid here.
        if self.mesh is None:
            self.mesh = RayCaster.meshes[self.scanner.cfg.mesh_prim_paths[0]]
        hits, _, normals, _ = raycast_mesh(
            starts, self.ray_directions, mesh=self.mesh,
            max_dist=3.0, return_normal=True,
        )
        valid = torch.isfinite(hits).all(-1) & torch.isfinite(normals).all(-1) & (normals[..., 2].abs() > .5)
        z = torch.where(valid, hits[..., 2], 0.)
        self.ground = z[..., 13]
        self.ground_valid = valid[..., 13]
        self.footprint_z = z[..., 13:]
        self.footprint_valid = valid[..., 13:].all(-1)
        jumps = (z[..., 1:13] - z[..., :12]).abs() > .025
        jumps &= valid[..., 1:13] & valid[..., :12]
        found = jumps.any(-1)
        index = jumps.long().argmax(-1)
        gather = lambda values, idx: values.gather(-1, idx[..., None]).squeeze(-1)
        before, after = gather(z, index), gather(z, index + 1)
        distance = (self.distances[index] + self.distances[index + 1]) * .5
        candidate_edge = self.center[..., :2] + direction[:, None, :] * distance[..., None]
        # The next discontinuity bounds the target tread. The horizon is a
        # conservative bound when the next riser lies outside the ray strip.
        sample_index = self.sample_index
        later = jumps & (sample_index > index[..., None])
        next_index = torch.where(later, sample_index, 12).amin(-1)
        end_distance = torch.where(next_index < 12,
                                   self.distances[next_index.clamp(max=11)] + .015, .30)
        width = end_distance - distance
        candidate_target = candidate_edge + direction[:, None, :] * (width * .5)[..., None]
        planned_swing = (self.progress > 0) & (self.progress < 1) & self.active[:, None]
        entering = planned_swing & ~self.was_swing
        self.locked &= ~entering
        self.launch = torch.where(entering & self.ground_valid, self.ground, self.launch)
        # A target is latched in WORLD coordinates for the rest of this swing.
        # This prevents the edge/landing target from moving with the foot.
        acquire = planned_swing & ~self.locked & found & (width > .12)
        self.edge = torch.where(acquire[..., None], candidate_edge, self.edge)
        self.direction = torch.where(acquire[..., None], direction[:, None, :], self.direction)
        self.top = torch.where(acquire, after, self.top)
        self.launch = torch.where(acquire, before, self.launch)
        self.target = torch.where(acquire[..., None], candidate_target, self.target)
        self.locked |= acquire
        self.was_swing.copy_(planned_swing)
        self.edge_distance = ((self.center[..., :2] - self.edge) * self.direction).sum(-1)
        self.near_edge = self.locked & (self.edge_distance.abs() < .06)
        self.edge_safe_height = torch.maximum(self.launch, self.top) + .025
        # A load-weighted support surface, not an equal average of four feet.
        # Smooth terrain transitions; hold the last reference during flight.
        weights = self.force[..., 2].clamp(0, 250) * self.contact * self.ground_valid
        total = weights.sum(-1)
        support = (weights * self.ground).sum(-1) / total.clamp_min(1e-6)
        # The four measured foot support surfaces are already available. They
        # also provide the initial airborne fallback without a second grid scan.
        scan = self.ground
        finite_scan = self.ground_valid
        fallback = torch.nanmedian(torch.where(finite_scan, scan, float("nan")), dim=-1).values
        available = (total > 1.) | finite_scan.any(-1)
        fallback = torch.nan_to_num(fallback)
        new = torch.where(total > 1., support, fallback)
        filtered = self.reference + min(env.step_dt / .12, 1.) * (new - self.reference)
        self.reference = torch.where(self.reference_valid,
                                     torch.where(total > 1., filtered, self.reference), new)
        self.reference_valid |= available



class _Dog2StairTerm(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        key = "_dog2_stair_geometry"
        if not hasattr(env, key):
            setattr(env, key, _StairGeometry(env, **cfg.params))
        self.geometry = getattr(env, key)

    def reset(self, env_ids=None):
        self.geometry.reset(env_ids)

    def __call__(self, env, asset_cfg, sensor_cfg, terrain_sensor_cfg, command_name, foot_offsets):
        self.geometry.update()
        return self.penalty(self.geometry)


class Dog2DynamicSwing(_Dog2StairTerm):
    def penalty(self, g):
        profile = torch.sin(torch.pi * g.progress).clamp_min(0)
        # Interpolate launch -> tread; add 4 cm over the higher surface at peak.
        destination = torch.where(g.locked, g.top, g.launch)
        reference = g.launch + (destination - g.launch) * g.progress
        lift = (destination - g.launch).abs() * .5 + .04
        target = reference + lift * profile
        # Do not ask the foot to descend below the edge term's safety height
        # while it is still crossing the riser, especially late in swing.
        target = torch.where(g.near_edge, torch.maximum(target, g.edge_safe_height), target)
        error = (target - g.sole) / .06
        asymmetric = error.clamp_min(0).square() + .2 * error.clamp_max(0).square()
        return (asymmetric.clamp(max=9) * g.swing * g.ground_valid * g.active[:, None]).mean(-1)


class Dog2EdgeClearance(_Dog2StairTerm):
    def penalty(self, g):
        near = g.near_edge
        deficit = ((g.edge_safe_height - g.sole).clamp_min(0) / .04).square().clamp(max=9)
        impact = ((g.force[..., :2].norm(dim=-1) - 25).clamp_min(0) / 50).square().clamp(max=9)
        return ((deficit + impact) * near * g.swing * g.active[:, None]).mean(-1)


class Dog2LandingPosition(_Dog2StairTerm):
    def penalty(self, g):
        position = ((g.center[..., :2] - g.target) / .12).square().sum(-1).clamp(max=9)
        span = g.footprint_z.amax(-1) - g.footprint_z.amin(-1)
        edge = (span > .025).float()
        speed = (g.velocity[..., :2].square().sum(-1) / .2**2).clamp(max=9)
        # Touchdown may occur just before or after phase boundary. The latched
        # target survives this event; no pressure to land during mid-swing.
        event = g.touchdown & g.locked & g.footprint_valid
        return ((position + edge + .2 * speed) * event * g.active[:, None]).mean(-1)


class Dog2StanceStability(_Dog2StairTerm):
    def penalty(self, g):
        slip = (g.velocity[..., :2].square().sum(-1) / .2**2).clamp(max=9)
        force = ((g.force[..., 2] - 250).clamp_min(0) / 100).square().clamp(max=9)
        # Also covers flat patches: replaces generic gait_contact/feet_slide.
        error = (~g.contact).float() + g.contact * (slip + .1 * force)
        return (error * g.stance * g.active[:, None]).mean(-1)


class Dog2DynamicBaseHeight(_Dog2StairTerm):
    def penalty(self, g):
        height = g.asset.data.root_pos_w[:, 2] - g.reference
        # A tolerance band allows a natural pitch/body transition on stairs.
        return ((height - .36).abs() - .025).clamp_min(0).square() * g.reference_valid


def configure_stair_rewards(cfg):
    """Replace overlapping terms only in Dog2 stairs; leave flat task intact."""
    for name in ("swing_clearance", "stair_landing", "gait_contact", "feet_slide", "base_height_l2", "feet_stumble", "contact_forces"):
        setattr(cfg.rewards, name, None)
    common = dict(
        asset_cfg=SceneEntityCfg("robot", body_names=list(FOOT_NAMES), preserve_order=True),
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES), preserve_order=True),
        terrain_sensor_cfg=SceneEntityCfg("height_scanner"),
        command_name="base_velocity", foot_offsets=FOOT_OFFSETS,
    )
    for name, function, weight in (
        ("stair_dynamic_swing", Dog2DynamicSwing, -5.),
        ("stair_edge_clearance", Dog2EdgeClearance, -5.),
        ("stair_landing_position", Dog2LandingPosition, -3.),
        ("stair_stance_stability", Dog2StanceStability, -2.),
        ("stair_dynamic_base_height", Dog2DynamicBaseHeight, -55.),
    ):
        setattr(cfg.rewards, name, RewardTermCfg(func=function, weight=weight, params=dict(common)))
