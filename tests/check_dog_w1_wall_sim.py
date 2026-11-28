"""Isaac Lab smoke check: isaaclab.sh -p tests/check_dog_w1_wall_sim.py --headless."""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/robot_lab"))
import robot_lab.tasks
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from robot_lab.him import HIMVecEnvWrapper, align_him_critic_observations

try:
    for terrain in ("Wall",):
        task = f"RobotLab-Isaac-Velocity-{terrain}-ATDog-DogW1-v0"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_him_cfg_entry_point")
        height = cfg.wall_height
        assert cfg.scene.wall.spawn.size == (cfg.wall_thickness, cfg.wall_length, height)
        assert cfg.scene.wall.init_state.pos == (cfg.wall_x, 0., height / 2)
        assert cfg.rewards.phase_swing_clearance is None
        assert cfg.events.randomize_apply_external_force_torque is None
        assert cfg.terminations.illegal_contact is None
        assert cfg.rewards.wall_response_lift.weight == 3.0
        assert cfg.scene.wall_contact_base.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Wall"]
        cfg.from_dict(cfg.to_dict())
        cfg.scene.num_envs = 4
        cfg.sim.device = args.device
        cfg.observations.policy.history_length = 6
        cfg.observations.policy.flatten_history_dim = True
        align_him_critic_observations(cfg, agent)
        env = HIMVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=100., num_one_step_obs=59,
                              history_order=agent.policy.history_order,
                              estimator_target_term_names=agent.policy.estimator_target_term_names)
        try:
            assert (env.num_obs, env.num_privileged_obs, env.num_actions) == (354, 62, 16)
            for _ in range(20):
                _, rewards, _, _ = env.step(torch.zeros(4, 16, device=env.device))
                assert torch.isfinite(rewards).all()
                state = env.unwrapped.reward_manager.get_term_cfg("wall_response_hip").func
                assert torch.isfinite(state.values["slip"]).all()
                assert env.unwrapped.scene.sensors["wall_contact_base"].data.force_matrix_w is not None
            print("WALL_OK", terrain)
        finally:
            env.close()
finally:
    app.close()
