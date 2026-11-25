"""dog_w1 asset/registration checks that run without Isaac Sim."""

import ast
import copy
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/robot_lab/robot_lab"
TASK = PACKAGE / "tasks/manager_based/locomotion/velocity/config/wheeled/dog_w1"
ASSET = ROOT / "source/robot_lab/data/Robots/atdog/dog_w1"


class Config(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def copy(self):
        return copy.deepcopy(self)


class DogW1ConfigTest(unittest.TestCase):
    def asset_cfg(self):
        tree = ast.parse((PACKAGE / "assets/atdog.py").read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        sim = SimpleNamespace(**{name: Config for name in (
            "UrdfFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg")})
        sim.UrdfConverterCfg = SimpleNamespace(JointDriveCfg=Config)
        Config.PDGainsCfg = Config
        Config.InitialStateCfg = Config
        namespace = {"sim_utils": sim, "ArticulationCfg": Config, "ISAACLAB_ASSETS_DATA_DIR": "data",
                     "DCMotorCfg": Config, "DelayedPDActuatorCfg": Config, "ImplicitActuatorCfg": Config, "FrictionDelayedPDActuatorCfg": Config, "FrictionImplicitActuatorCfg": Config}
        exec(compile(tree, "atdog.py", "exec"), namespace)
        return namespace

    def test_initial_state_and_actuators_cover_all_urdf_joints(self):
        configs = self.asset_cfg()
        cfg = configs["AT_DOG_W1_CFG"]
        urdf = ET.parse(ASSET / "urdf/dog_w1.urdf").getroot()
        joints = urdf.findall("joint")
        self.assertEqual(len(joints), 16)
        self.assertTrue(cfg.spawn.asset_path.endswith("dog_w1/urdf/dog_w1.urdf"))
        self.assertEqual(cfg.init_state.pos, (0.0, 0.0, 0.35))
        for joint in joints:
            name = joint.get("name")
            values = [value for pattern, value in cfg.init_state.joint_pos.items() if re.fullmatch(pattern, name)]
            self.assertEqual(len(values), 1, name)
            motors = [key for key, motor in cfg.actuators.items()
                      if any(re.fullmatch(pattern, name) for pattern in motor.joint_names_expr)]
            self.assertEqual(len(motors), 1, name)
            if joint.get("type") == "continuous":
                self.assertEqual(motors, ["wheels"])
            else:
                limits = joint.find("limit")
                self.assertLessEqual(float(limits.get("lower")), values[0])
                self.assertGreaterEqual(float(limits.get("upper")), values[0])
        self.assertEqual(cfg.actuators["wheels"].stiffness, 0.0)
        self.assertNotIn("wheels", configs["AT_DOG2_CFG"].actuators)
        self.assertNotIn(".*_foot_joint", configs["AT_DOG2_CFG"].init_state.joint_pos)

    def test_initialization_matches_dog2_legs_and_atdog_wheels(self):
        configs = self.asset_cfg()
        cfg, dog2, atdog = (configs[name] for name in ("AT_DOG_W1_CFG", "AT_DOG2_CFG", "AT_DOG_CFG"))
        for pattern, value in dog2.init_state.joint_pos.items():
            self.assertEqual(cfg.init_state.joint_pos[pattern], value)
        self.assertEqual(cfg.init_state.joint_pos[".*_foot_joint"], atdog.init_state.joint_pos[".*_foot_joint"])
        self.assertEqual(cfg.init_state.joint_vel, dog2.init_state.joint_vel)
        self.assertEqual(vars(cfg.actuators["wheels"]), vars(atdog.actuators["wheels"]))
        for key, actuator in dog2.actuators.items():
            expected = {**vars(actuator), "friction": 0.0}
            self.assertEqual(vars(cfg.actuators[key]), expected)
            self.assertIsNot(cfg.actuators[key], actuator)

    def test_tasks_and_runners_have_no_robot_task_inheritance(self):
        for path in TASK.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse("config.wheeled.atdog" in (node.module or "")
                                     or "config.quadruped.atdog" in (node.module or ""), str(path))
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        self.assertFalse(ast.unparse(base).startswith("ATDog"), (path, node.name))

    def test_phase_first_actor_and_critic_teacher_slices(self):
        widths = {"phase": 2, "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,
                  "velocity_commands": 3, "joint_pos": 16, "joint_vel": 16, "actions": 16}
        expected_actor = ["phase", "base_ang_vel", "projected_gravity", "velocity_commands",
                          "joint_pos", "joint_vel", "actions"]
        expected_critic = ["base_lin_vel", *expected_actor[1:], "phase"]
        him_tree = ast.parse((TASK / "agents/rsl_rl_him_cfg.py").read_text())
        policy_cfg = next(n for n in him_tree.body if isinstance(n, ast.ClassDef)
                          and n.name == "RslRlHimActorCriticCfg")
        policy_values = {n.targets[0].id: ast.literal_eval(n.value) for n in policy_cfg.body if isinstance(n, ast.Assign)}
        self.assertEqual(policy_values["num_one_step_obs"], 59)
        self.assertEqual(policy_values["history_length"], 6)
        self.assertEqual(policy_values["history_order"], "oldest_to_latest")
        self.assertEqual(policy_values["actor_current_frame"], "last")
        # Exercise the real wrapper's metadata-based target selection without torch.
        wrapper = ast.parse((PACKAGE / "him/him_vec_env_wrapper.py").read_text())
        cls = next(n for n in wrapper.body if isinstance(n, ast.ClassDef) and n.name == "HIMVecEnvWrapper")
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name in ("_infer_teacher_target_slices", "_term_slice")]
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "him_vec_env_wrapper.py", "exec"), namespace)
        for terrain in ("flat", "rough", "slope", "sand", "stairs"):
            tree = ast.parse((TASK / f"{terrain}_env_cfg.py").read_text())
            layouts = []
            for suffix, expected, total in (("PolicyCfg", expected_actor, 59), ("CriticCfg", expected_critic, 62)):
                group = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith(suffix))
                names = [n.targets[0].id for n in group.body if isinstance(n, ast.Assign)
                         and n.targets[0].id in widths]
                self.assertEqual(names, expected)
                cursor, layout = 0, {}
                for name in names:
                    layout[name] = slice(cursor, cursor + widths[name])
                    cursor += widths[name]
                self.assertEqual(cursor, total)
                layouts.append(layout)
            live = namespace["HIMVecEnvWrapper"]()
            live.critic_term_slices = layouts[1]
            live.teacher_target_terms = policy_values["estimator_target_term_names"]
            live.explicit_teacher_target_terms = True
            live.num_one_step_obs = 59
            slices = live._infer_teacher_target_slices()
            self.assertEqual(slices, ((3, 6), (6, 9), (12, 28), (28, 44), (44, 60), (0, 3)))
            self.assertEqual(sum(stop - start for start, stop in slices), 57)
            self.assertEqual(live._term_slice(live.critic_term_slices, "base_lin_vel"), (0, 3))
            self.assertNotIn("phase", live.teacher_target_terms)
            self.assertNotIn("velocity_commands", live.teacher_target_terms)

    def test_urdf_meshes_exist(self):
        urdf = ET.parse(ASSET / "urdf/dog_w1.urdf").getroot()
        for mesh in urdf.findall(".//mesh"):
            relative = mesh.get("filename").removeprefix("package://dog_w1/")
            self.assertTrue((ASSET / relative).is_file(), relative)

    def test_registration_entry_points_exist(self):
        registrations = []
        tree = ast.parse((TASK / "__init__.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                values = {kw.arg: kw.value for kw in call.keywords}
                registrations.append(ast.literal_eval(values["id"]))
                for value in values["kwargs"].values:
                    target = "".join(part.value if isinstance(part, ast.Constant) else
                                     "agents" if ast.unparse(part.value) == "agents.__name__" else "task"
                                     for part in value.values)
                    module, cls = target.split(":")
                    path = TASK / (module.removeprefix("task.").replace(".", "/") + ".py")
                    definitions = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
                    self.assertIn(cls, definitions)
        self.assertEqual(set(registrations), {
            f"RobotLab-Isaac-Velocity-{terrain}-ATDog-DogW1-v0"
            for terrain in ("Flat", "Rough", "Slope", "Sand", "Stairs", "Tires")})

    def test_wheel_angles_masked_after_reordering_joints(self):
        tree = ast.parse((TASK / "observations.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "joint_pos_rel_without_wheel")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "observations.py", "exec"), namespace)
        data = SimpleNamespace(joint_pos=np.arange(1., 17.)[None, :], default_joint_pos=np.ones((1, 16)))
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        # Articulation IDs interleave wheels; selected columns put wheels last.
        order = [i for i in range(16) if i % 4 != 3] + [3, 7, 11, 15]
        result = namespace[function.name](env, SimpleNamespace(name="robot", joint_ids=order),
                                          SimpleNamespace(joint_ids=[3, 7, 11, 15]))
        np.testing.assert_array_equal(result[0, :12], np.array(order[:12]))
        np.testing.assert_array_equal(result[0, 12:], 0.)
        np.testing.assert_array_equal(data.joint_pos, np.arange(1., 17.)[None, :])


if __name__ == "__main__":
    unittest.main()
