"""Isolated Dog2 stair training/evaluation stage; does not edit default tasks."""
import argparse
import copy
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--mode', choices=('train', 'eval'), required=True)
parser.add_argument('--num_envs', type=int, default=5500)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--low', type=float, default=.06)
parser.add_argument('--high', type=float, default=.11)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--profile', choices=('balanced', 'clearance', 'support'), default='balanced')
args = parser.parse_args()
app = AppLauncher(headless=True).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.terrains import TerrainGenerator
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry
import robot_lab.tasks  # noqa: F401
from robot_lab.him import HIMVecEnvWrapper, HIMOnPolicyRunner


class StageTerrain(TerrainGenerator):
    """Honor each stair's configured range, preserving flat patch metadata."""
    def _generate_curriculum_terrains(self):
        configs = list(self.cfg.sub_terrains.values())
        proportions = np.array([cfg.proportion for cfg in configs])
        proportions = np.cumsum(proportions / proportions.sum())
        flat = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=bool)
        for col in range(self.cfg.num_cols):
            kind = int(np.searchsorted(proportions, col / self.cfg.num_cols + .001))
            for row in range(self.cfg.num_rows):
                cfg = copy.deepcopy(configs[kind])
                difficulty = row / max(self.cfg.num_rows - 1, 1)
                if hasattr(cfg, 'step_height_range'):
                    lo, hi = cfg.step_height_range
                    height = lo + (hi - lo) * difficulty
                    cfg.step_height_range = (height, height)
                flat[row, col] = kind == 0
                mesh, origin = self._get_terrain_mesh(difficulty, cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)
        self.cfg.flat_patch_mask = flat.reshape(-1).tolist()


def stable_curriculum(env, env_ids):
    """Require two safe traversals; retain the highest row instead of wrapping."""
    terrain = env.scene.terrain
    command = env.command_manager.get_term('base_velocity')
    eligible = command._had_translation[env_ids] & command._start_valid[env_ids]
    failed = env.termination_manager.terminated[env_ids]
    success = eligible & ~failed & (command.metrics['stair_progress'][env_ids] >= 1.2)
    success &= command.metrics['stair_height_change'][env_ids] >= .12
    if not hasattr(env, '_stage_success_streak'):
        env._stage_success_streak = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    streak = env._stage_success_streak
    streak[env_ids] = torch.where(success, streak[env_ids] + 1, 0)
    up = success & (streak[env_ids] >= 2) & (terrain.terrain_levels[env_ids] < terrain.max_terrain_level - 1)
    down = eligible & (failed | (command.metrics['stair_progress'][env_ids] < .6))
    terrain.update_env_origins(env_ids, up, down)
    streak[env_ids] = torch.where(up, 0, streak[env_ids])
    return terrain.terrain_levels.float().mean()


def main():
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    task = 'RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_him_cfg_entry_point')
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.log_dir = str(output)
    cfg.observations.policy.history_length = int(agent.policy.history_length)
    cfg.observations.policy.flatten_history_dim = True
    terrain = copy.deepcopy(cfg.scene.terrain.terrain_generator)
    cfg.scene.terrain.terrain_generator = terrain
    terrain.class_type = StageTerrain
    terrain.curriculum = True
    terrain.use_cache = False
    for sub in terrain.sub_terrains.values():
        if hasattr(sub, 'step_height_range'):
            sub.step_height_range = (args.low, args.high) if args.mode == 'train' else (.11, .11)
    if args.mode == 'eval':
        # Ascending-only: outward from the inverted pyramid's low platform.
        for key, sub in terrain.sub_terrains.items():
            sub.proportion = float(key == 'stairs')
        terrain.num_rows = 1
        terrain.num_cols = 8
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.curriculum.terrain_levels = None
        cfg.commands.base_velocity.rel_standing_envs = 0.
        cfg.commands.base_velocity.stair_speed_range = (.3, .5)
    else:
        for key, sub in terrain.sub_terrains.items():
            sub.proportion = {'flat': .2, 'stairs': .65, 'stairs2': .15}[key]
        cfg.curriculum.terrain_levels.func = stable_curriculum
        cfg.commands.base_velocity.stair_speed_range = (.25, .6)
        cfg.commands.base_velocity.success_distance = 1.2
        cfg.commands.base_velocity.min_height_change = .12
        # The joint acceleration cost dominates current penalties. Ease it
        # moderately while keeping action-rate and smoothness regularization.
        cfg.rewards.joint_acc_l2.weight *= .75
        cfg.rewards.stair_dynamic_swing.weight = -3.
        cfg.rewards.stair_edge_clearance.weight = -5.
        cfg.rewards.stair_landing_position.weight = -3.
        cfg.rewards.stair_stance_stability.weight = -1.5
        if args.profile == 'clearance':
            cfg.rewards.stair_dynamic_swing.weight = -4.
            cfg.rewards.stair_edge_clearance.weight = -6.
        elif args.profile == 'support':
            cfg.rewards.stair_stance_stability.weight = -2.
            cfg.rewards.stair_dynamic_base_height.weight = -40.
    dump_yaml(str(output / 'params/env.yaml'), cfg)
    dump_yaml(str(output / 'params/agent.yaml'), agent)
    (output / 'stage.json').write_text(json.dumps(vars(args), indent=2))
    env = gym.make(task, cfg=cfg)
    wrapped = HIMVecEnvWrapper(
        env, clip_actions=agent.clip_actions,
        num_one_step_obs=agent.policy.num_one_step_obs,
        history_order=agent.policy.history_order,
        estimator_target_term_names=agent.policy.estimator_target_term_names,
    )
    runner = HIMOnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(output) if args.mode == 'train' else None, device=agent.device)
    runner.load(args.checkpoint, inference=args.mode == 'eval')
    if args.mode == 'train':
        runner.learn(args.iterations, init_at_random_ep_len=False)
    else:
        raw = env.unwrapped
        command = raw.command_manager.get_term('base_velocity')
        completed = []
        reset = raw._reset_idx
        def capture(ids):
            if ids.numel():
                progress = command.metrics['stair_progress'][ids]
                rise = command.metrics['stair_height_change'][ids]
                failed = raw.termination_manager.terminated[ids]
                length = raw.episode_length_buf[ids]
                # A pass needs >=4 treads horizontally and >=2 risers in
                # height, surviving to timeout; early falls are always failures.
                success = ~failed & (progress >= 1.2) & (rise >= .22)
                completed.extend(torch.stack((success, failed, progress, rise, length), -1).cpu().tolist())
            return reset(ids)
        raw._reset_idx = capture
        policy = runner.get_inference_policy()
        obs = wrapped.get_observations()
        with torch.inference_mode():
            for _ in range(3 * raw.max_episode_length):
                obs, _, _, _ = wrapped.step(policy(obs))
        samples = np.array(completed)
        if not len(samples):
            raise RuntimeError('No completed evaluation episodes')
        result = dict(seed=args.seed, height=.11, episodes=len(samples),
                      success_rate=float(samples[:, 0].mean()), fall_rate=float(samples[:, 1].mean()),
                      progress=float(samples[:, 2].mean()), height_change=float(samples[:, 3].mean()),
                      checkpoint=args.checkpoint, speed_range=[.3, .5],
                      note='Ascending, all signed XY directions, trained randomization/noise retained')
        (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
