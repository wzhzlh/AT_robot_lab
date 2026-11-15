"""Read-only rollout profiling with nested CPU/CUDA inclusive timings."""
import argparse
import json
from pathlib import Path
import time
import faulthandler
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=5500)
parser.add_argument('--warmup', type=int, default=96)
parser.add_argument('--steps', type=int, default=240)
parser.add_argument('--stream-actuators', action='store_true')
parser.add_argument('--no-instrument', action='store_true')
args = parser.parse_args()
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
app = AppLauncher(headless=True).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
import robot_lab.tasks  # noqa: F401
from robot_lab.him import HIMOnPolicyRunner, HIMVecEnvWrapper, align_him_critic_observations


class Timings:
    def __init__(self):
        self.records = {}
        self.enabled = False

    def wrap(self, obj, name, label):
        if args.no_instrument:
            return
        original = getattr(obj, name)
        def timed(*positional, **keywords):
            if not self.enabled:
                return original(*positional, **keywords)
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            wall = time.perf_counter()
            try:
                return original(*positional, **keywords)
            finally:
                elapsed = time.perf_counter() - wall
                stop.record()
                self.records.setdefault(label, []).append((elapsed, start, stop))
        if hasattr(original, 'reset'):
            timed.reset = original.reset
        setattr(obj, name, timed)

    def result(self):
        torch.cuda.synchronize()
        return {label: dict(calls=len(records),
                            cpu_wall_ms=sum(x[0] for x in records) * 1000,
                            cuda_timeline_ms=sum(x[1].elapsed_time(x[2]) for x in records))
                for label, records in self.records.items()}


def main():
    task = 'RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_him_cfg_entry_point')
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed
    cfg.observations.policy.history_length = agent.policy.history_length
    cfg.observations.policy.flatten_history_dim = True
    align_him_critic_observations(cfg, agent)
    from isaaclab.actuators import DelayedPDActuator
    from robot_lab.actuators import StreamDelayedPDActuator
    # Explicit baseline override even when the task enables optimized buffers.
    for actuator in cfg.scene.robot.actuators.values():
        actuator.class_type = StreamDelayedPDActuator if args.stream_actuators else DelayedPDActuator
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    wrapped = HIMVecEnvWrapper(env, clip_actions=agent.clip_actions,
                              num_one_step_obs=agent.policy.num_one_step_obs,
                              history_order=agent.policy.history_order,
                              estimator_target_term_names=agent.policy.estimator_target_term_names)
    runner = HIMOnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    print('PROFILE: runner ready', flush=True)
    runner.load(args.checkpoint, inference=True)
    print('PROFILE: checkpoint loaded', flush=True)
    model = runner.alg.actor_critic
    model.train()  # Sample the same action distribution used in collection.
    obs = wrapped.get_observations()
    timer = Timings()
    for obj, method, label in (
        (raw, 'step', 'env.total'), (raw.sim, 'step', 'physics.sim_step'),
        (raw.scene, 'write_data_to_sim', 'physics.write'),
        (raw.scene, 'update', 'scene.update'),
        (raw.action_manager, 'apply_action', 'action.apply'),
        (raw.action_manager, 'process_action', 'action.process'),
        (raw.reward_manager, 'compute', 'reward.total'),
        (raw.termination_manager, 'compute', 'termination.compute'),
        (raw.observation_manager, 'compute', 'observation.compute'),
        (raw.observation_manager, 'compute_group', 'observation.group'),
        (raw.command_manager, 'compute', 'command.compute'),
        (raw, '_reset_idx', 'reset.total'),
        (raw.event_manager, 'apply', 'events.apply'),
        (raw.curriculum_manager, 'compute', 'curriculum.compute'),
        (raw.scene, 'reset', 'reset.scene'),
    ):
        timer.wrap(obj, method, label)
    for name, sensor in raw.scene.sensors.items():
        timer.wrap(sensor, '_update_buffers_impl', f'sensor.{name}')
    robot = raw.scene['robot']
    timer.wrap(robot, 'write_data_to_sim', 'robot.write')
    timer.wrap(robot, '_apply_actuator_model', 'robot.actuator_model')
    for name, actuator in robot.actuators.items():
        timer.wrap(actuator, 'compute', f'actuator.{name}')
    for method in ('set_dof_actuation_forces', 'set_dof_position_targets', 'set_dof_velocity_targets'):
        timer.wrap(robot.root_physx_view, method, 'physx.' + method)
    for name, term_cfg in zip(raw.reward_manager._term_names, raw.reward_manager._term_cfgs):
        timer.wrap(term_cfg, 'func', f'reward.{name}')
    if hasattr(raw, '_dog2_stair_geometry'):
        timer.wrap(raw._dog2_stair_geometry, 'update', 'reward.stair_geometry')
    for name in ('observation', 'reward', 'command', 'termination', 'event', 'action'):
        timer.wrap(getattr(raw, name + '_manager'), 'reset', 'reset.' + name)
    with torch.inference_mode():
        for _ in range(args.warmup):
            obs, _, _, _ = wrapped.step(model.act(obs))
        torch.cuda.synchronize()
        timer.enabled = True
        wall = time.perf_counter()
        for _ in range(args.steps):
            obs, _, _, _ = wrapped.step(model.act(obs))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - wall
    result = dict(num_envs=args.num_envs, stream_actuators=args.stream_actuators, steps=args.steps, warmup=args.warmup,
                  wall_seconds=elapsed, checkpoint=args.checkpoint,
                  timings=timer.result(),
                  note='Inclusive nested timings; do not sum parent and child. CUDA timeline includes host launch gaps. Instrumentation adds overhead; no optimizer updates.')
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    app.close()
