"""Isaac Lab smoke checks for the real ATDog2 config/reward/observation wiring.

Run with isaaclab.sh -p tests/check_atdog2_gait_sim.py --headless.
The script uses four flat environments; it does not train a policy.
"""

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--terrain", choices=("flat", "stairs"), default="flat")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.util
import json
import os
import tempfile
import traceback
from types import SimpleNamespace
import unittest

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/robot_lab"))
import robot_lab
from robot_lab.gait import TrotClock, TrotClockCfg
from robot_lab.him import HIMActorCritic, HIMVecEnvWrapper, align_him_critic_observations
from robot_lab.him.exporter import export_him_deployment_metadata, export_him_policy_as_jit, export_him_policy_as_onnx
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.flat_env_cfg import (
    ATDogDog2FlatDomainRandEnvCfg, ATDogDog2FlatEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.rough_env_cfg import ATDogDog2RoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.stairs_env_cfg import ATDogDog2StairsEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.slope_env_cfg import ATDogDog2SlopeEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.sand_env_cfg import ATDogDog2SandEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.bar_env_cfg import ATDogDog2BarEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.atdog.dog2.tires_env_cfg import ATDogDog2TiresEnvCfg


def main():
    # Existing regression suites, including teacher/student scaling and histories.
    if args.terrain == "flat":
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        assert result.wasSuccessful()
    expected_names = ["phase", "base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    for cls in (ATDogDog2FlatEnvCfg, ATDogDog2FlatDomainRandEnvCfg, ATDogDog2RoughEnvCfg,
                ATDogDog2StairsEnvCfg, ATDogDog2SlopeEnvCfg, ATDogDog2SandEnvCfg,
                ATDogDog2BarEnvCfg, ATDogDog2TiresEnvCfg):
        cfg = cls()
        # Hydra restores nested configs through from_dict before launching training.
        cfg.from_dict(cfg.to_dict())
        cfg.from_dict({"commands": {"base_velocity": {"gait": {"cycle_time": 0.6}}}})
        assert cfg.commands.base_velocity.gait.cycle_time == 0.6
        cfg.from_dict({"commands": {"base_velocity": {"gait": {"cycle_time": TrotClockCfg().cycle_time}}}})
        names = [name for name, term in vars(cfg.observations.policy).items() if hasattr(term, "func")]
        assert names == expected_names, (cls.__name__, names)
        assert cfg.rewards.gait_contact.weight < 0
        assert cfg.rewards.swing_clearance.weight < 0
        for removed in ("periodic_foot_lift", "feet_gait", "diagonal_trot_contact_pattern", "feet_height",
                        "feet_height_body", "feet_height_body_exp", "feet_contact_without_cmd", "feet_air_time"):
            assert getattr(cfg.rewards, removed) is None, (cls.__name__, removed)
        assert cfg.commands.base_velocity.gait == TrotClockCfg()
        print("CONFIG_OK", cls.__name__)

    cfg = ATDogDog2FlatEnvCfg() if args.terrain == "flat" else ATDogDog2StairsEnvCfg()
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 2.5
    cfg.seed = 7
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 2
        cfg.scene.terrain.max_init_terrain_level = 1
    cfg.sim.device = args.device
    cfg.observations.policy.history_length = 6
    align_him_critic_observations(cfg, SimpleNamespace(class_name="HIMOnPolicyRunner"))
    task = "RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0" if args.terrain == "flat" else "RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0"
    env = gym.make(task, cfg=cfg)
    wrapper = HIMVecEnvWrapper(env, clip_actions=1.0, num_one_step_obs=47, history_order="oldest_to_latest")
    try:
        raw = env.unwrapped
        names = list(wrapper.policy_one_step_term_slices)
        assert names == expected_names, names
        assert (wrapper.num_one_step_obs, wrapper.num_obs, wrapper.history_length) == (47, 282, 6)
        assert [sl.start for sl in wrapper.policy_one_step_term_slices.values()] == [0, 2, 5, 8, 11, 23, 35]
        assert [sl.stop for sl in wrapper.policy_one_step_term_slices.values()] == [2, 5, 8, 11, 23, 35, 47]
        term = raw.command_manager.get_term("base_velocity")
        # Drive translation, pure rotation and stopping independently of random commands.
        term.is_standing_env[:] = False
        with torch.inference_mode():
            for step in range(50):
                term.vel_command_b[:] = 0
                if step < 20:
                    term.vel_command_b[:, 0] = 0.4
                elif step < 35:
                    term.vel_command_b[:, 2] = 0.4
                obs, reward, dones, extras = wrapper.step(torch.zeros(4, 12, device=raw.device))
                assert obs.shape == (4, 282)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                # Latest frame is last; compare with the live command clock.
                torch.testing.assert_close(obs[:, -47:-45], term.gait_clock.observation())
                values = raw.reward_manager._step_reward
                assert torch.isfinite(values).all()
            assert not term.gait_clock.active.any()

        # Stationary support is a sustained state, independent of first-contact events.
        from robot_lab.tasks.manager_based.locomotion.velocity import mdp
        stand_cfg = raw.reward_manager.get_term_cfg("standing_support")
        support = mdp.standing_support(raw, **stand_cfg.params)
        assert torch.isfinite(support).all() and (support >= 0).all()
        if args.terrain == "flat":
            # Construct yaw rates to verify both magnitude and command gating.
            fake_asset = SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=torch.tensor(
                [[0., 0., 0.], [0., 0., 0.15], [0., 0., -0.15], [0., 0., 0.15]], device=raw.device
            )))
            fake_clock = SimpleNamespace(active=torch.tensor([False, False, False, True], device=raw.device))
            fake_env = SimpleNamespace(
                scene={"robot": fake_asset},
                command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(gait_clock=fake_clock)),
            )
            penalty = mdp.standing_yaw_drift(fake_env, "base_velocity", 0.15)
            torch.testing.assert_close(penalty, torch.tensor([0., 1., 1., 0.], device=raw.device))

        actor = HIMActorCritic(
            num_actor_obs=282, num_one_step_obs=47, num_critic_obs=wrapper.num_privileged_obs, num_actions=12,
            estimator_vel_slice=wrapper.default_estimator_vel_slice,
            estimator_target_slices=wrapper.default_estimator_target_slices,
            actor_current_frame="last",
        )
        with tempfile.TemporaryDirectory() as directory:
            export_him_policy_as_jit(actor, directory)
            export_him_policy_as_onnx(actor, directory)
            scripted = torch.jit.load(str(Path(directory) / "policy.pt"))
            sample = torch.randn(1, 282)
            with torch.inference_mode():
                torch.testing.assert_close(scripted(sample), actor.act_inference(sample))
            paths = export_him_deployment_metadata(actor, wrapper, directory)
            meta = json.loads(Path(paths["metadata_path"]).read_text())
            assert meta["gait_clock"]["encoding"] == ["sin", "cos"]
            assert meta["recommended_at_robot_lab_history"]["observations_history"] == [5, 4, 3, 2, 1, 0]
            assert meta["recommended_at_robot_lab_history"]["history_layout"] == "time_major_oldest_to_current"
            spec = importlib.util.spec_from_file_location("exported_gait_clock", paths["gait_clock_path"])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            deploy = module.TrotClock.from_metadata(1, "cpu", meta["gait_clock"])
            reference = TrotClock(1, "cpu", TrotClockCfg())
            for command in ([0.5, 0, 0], [0, 0, 0.4], [0, 0, 0], [0, 0, 0]):
                for _ in range(17):
                    values = torch.tensor([command], dtype=torch.float32)
                    deploy.update(values, 0.02)
                    reference.update(values, 0.02)
                    torch.testing.assert_close(deploy.observation(), reference.observation())
        print(f"SIM_OK ({args.terrain}): 47-D order, 282-D history, finite rewards, translation/yaw/stop, ONNX/JIT, metadata and deployment clock", flush=True)
    finally:
        wrapper.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    print("SIM_FAILED", flush=True)
    # Isaac shutdown may replace an active exception's exit status with zero.
    sys.stderr.flush()
    os._exit(1)
finally:
    app.close()
