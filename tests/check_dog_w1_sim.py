"""Run with isaaclab.sh -p tests/check_dog_w1_sim.py --headless.

Checks all five config/runner pairs, then steps four flat environments and checks
HIM estimator supervision and inference/export agreement. Does not train a policy.
"""

import argparse
from pathlib import Path
import sys
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/robot_lab"))
import robot_lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from robot_lab.him import HIMOnPolicyRunner, HIMVecEnvWrapper, align_him_critic_observations
from robot_lab.him.exporter import export_him_policy_as_jit, export_him_deployment_metadata


def main():
    expected_names = ["phase", "base_ang_vel", "projected_gravity", "velocity_commands",
                      "joint_pos", "joint_vel", "actions"]
    for terrain in ("Flat", "Rough", "Slope", "Sand", "Stairs"):
        task = f"RobotLab-Isaac-Velocity-{terrain}-ATDog-DogW1-v0"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_him_cfg_entry_point")
        cfg.from_dict(cfg.to_dict())
        align_him_critic_observations(cfg, agent)
        names = [name for name, value in vars(cfg.observations.policy).items() if hasattr(value, "func")]
        assert names == expected_names, (terrain, names)
        assert cfg.base_link_name == "base"
        assert len(cfg.actions.joint_pos.joint_names) == 12
        assert len(cfg.actions.joint_vel.joint_names) == 4
        assert cfg.rewards.feet_contact.func.__name__ == "PhaseContactReward"
        assert cfg.rewards.feet_contact_without_cmd.func.__name__ == "standing_contact"
        assert cfg.rewards.phase_swing_clearance.weight < 0
        assert cfg.rewards.feet_air_time is None
        if terrain == "Flat":
            assert cfg.rewards.feet_contact.params["centered"]
            assert cfg.rewards.lateral_velocity_error.weight < 0
            assert cfg.terminations.illegal_contact is not None
            assert cfg.terminations.bad_orientation.params["limit_angle"] == 1.0
            assert cfg.events.randomize_push_robot is None
            assert agent.algorithm.entropy_coef == 0.003
        for name, value in cfg.observations.policy.phase.params.items():
            assert cfg.rewards.feet_contact.params[name] == value
            assert cfg.rewards.phase_swing_clearance.params[name] == value
        assert cfg.observations.critic.phase is not None
        assert not cfg.observations.critic.enable_corruption
        for entry in ("rsl_rl_cfg_entry_point", "cusrl_cfg_entry_point"):
            load_cfg_from_registry(task, entry)
        print("CONFIG_OK", terrain)

    task = "RobotLab-Isaac-Velocity-Flat-ATDog-DogW1-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_him_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.observations.policy.history_length = agent.policy.history_length
    cfg.observations.policy.flatten_history_dim = True
    align_him_critic_observations(cfg, agent)
    raw_env = gym.make(task, cfg=cfg)
    env = HIMVecEnvWrapper(
        raw_env, clip_actions=100.0, num_one_step_obs=agent.policy.num_one_step_obs,
        history_order=agent.policy.history_order,
        estimator_target_term_names=agent.policy.estimator_target_term_names,
    )
    try:
        runner = HIMOnPolicyRunner(env, agent.to_dict(), device=args.device)
        model = runner.alg.actor_critic
        assert (env.num_obs, env.num_one_step_obs, env.num_privileged_obs, env.num_actions) == (354, 59, 62, 16)
        assert sum(stop - start for start, stop in model.target_slices) == 57
        for _ in range(10):
            obs = env.get_observations()
            with torch.inference_mode():
                actions = model.act_inference(obs)
            _, rewards, _, _ = env.step(actions)
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(env.get_critic_observations()).all()
            contact = raw_env.unwrapped.reward_manager.get_term_cfg("feet_contact").func
            swing = raw_env.unwrapped.reward_manager.get_term_cfg("phase_swing_clearance").func
            torch.testing.assert_close(contact.clock.phase, swing.clock.phase)
            assert contact.clock._state is swing.clock._state
        obs = env.get_observations()
        teacher, valid = env.get_estimator_next_critic_observations()
        losses = model.estimator.update(obs, teacher, valid_mask=valid)
        assert all(torch.isfinite(torch.tensor(value)) for value in losses)
        with tempfile.TemporaryDirectory() as output:
            export_him_policy_as_jit(model, output)
            export_him_deployment_metadata(model, env, output)
            scripted = torch.jit.load(str(Path(output) / "policy.pt"), map_location=env.device)
            with torch.inference_mode():
                torch.testing.assert_close(scripted(obs), model.act_inference(obs))
        print("HIM_OK: actor history=354, critic=62, actions=16, teacher=57")
    finally:
        env.close()


try:
    main()
finally:
    app.close()
