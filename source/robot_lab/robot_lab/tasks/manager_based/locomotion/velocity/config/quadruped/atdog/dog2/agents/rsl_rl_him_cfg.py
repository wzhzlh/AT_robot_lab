# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlHimActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "HIMActorCritic"
    num_one_step_obs = 47
    history_length = 6
    history_order = "oldest_to_latest"
    actor_current_frame = "last"
    estimator_target_term_names = [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "base_lin_vel",
    ]
    estimator_encoder_hidden_dims = [128, 64, 16]
    estimator_target_hidden_dims = [128, 64]
    estimator_learning_rate = 1.0e-3
    estimator_max_grad_norm = 10.0
    estimator_num_prototypes = 32
    estimator_temperature = 3.0
    estimator_vel_slice = None


@configclass
class RslRlHimPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    estimator_learning_rate = 1.0e-3
    estimator_velocity_loss_coef = 1.0
    estimator_swap_loss_coef = 0.1


@configclass
class ATDogDog2RoughHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_rough_him"
    policy = RslRlHimActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlHimPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ATDogDog2SandHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_sand_him"
    policy = RslRlHimActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlHimPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ATDogDog2SlopeHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog2_slope_him"
    policy = RslRlHimActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlHimPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ATDogDog2FlatHIMRunnerCfg(ATDogDog2RoughHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 48
        self.max_iterations = 1000
        self.save_interval = 50
        self.algorithm.learning_rate = 5.0e-4
        self.experiment_name = "atdog_dog2_flat_him"


@configclass
class ATDogDog2FlatDomainRandHIMRunnerCfg(ATDogDog2FlatHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.experiment_name = "atdog_dog2_flat_domain_rand_him"
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.003
        self.algorithm.schedule = "fixed"


@configclass
class ATDogDog2StairsHIMRunnerCfg(ATDogDog2FlatDomainRandHIMRunnerCfg):
    """Continue the shared phase-free policy through the stair stage."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000
        self.save_interval = 100
        self.experiment_name = "atdog_dog2_stairs_him"


@configclass
class ATDogDog2BarHIMRunnerCfg(ATDogDog2RoughHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "atdog_dog2_bar_him"


@configclass
class ATDogDog2TiresHIMRunnerCfg(ATDogDog2RoughHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "atdog_dog2_tires_him"
