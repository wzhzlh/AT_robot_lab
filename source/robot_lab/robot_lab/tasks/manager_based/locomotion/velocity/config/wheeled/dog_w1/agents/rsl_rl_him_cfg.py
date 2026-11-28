# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlHimActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "HIMActorCritic"
    num_one_step_obs = 59
    history_length = 6
    history_order = "oldest_to_latest"
    actor_current_frame = "last"
    # Same teacher features as dog2, with 16 joints/actions: 57 values.
    # Phase and velocity commands condition the actor but are excluded from the target.
    estimator_target_term_names = [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "base_lin_vel",
    ]
    estimator_encoder_hidden_dims = [128, 64, 16]
    estimator_target_hidden_dims = [128, 64]
    estimator_learning_rate = 1.0e-3
    estimator_max_grad_norm = 10.0
    estimator_num_prototypes = 32
    estimator_temperature = 3.0
    # The runner resolves velocity and teacher slices from the live critic layout.
    estimator_vel_slice = None


@configclass
class RslRlHimPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    estimator_learning_rate = 1.0e-3
    estimator_velocity_loss_coef = 1.0
    estimator_swap_loss_coef = 0.1


@configclass
class ATDogDogW1RoughHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog_w1_rough_him"
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
class ATDogDogW1SlopeHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog_w1_slope_him"
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
class ATDogDogW1StairsHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog_w1_stairs_him"
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
class ATDogDogW1SandHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog_w1_sand_him"
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
class ATDogDogW1FlatHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "atdog_dog_w1_flat_him"
    policy = RslRlHimActorCriticCfg(
        init_noise_std=0.6,
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
        entropy_coef=0.003,
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
class ATDogDogW1TiresHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "HIMOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "atdog_dog_w1_tires_him"
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
class ATDogDogW1WallHIMRunnerCfg(ATDogDogW1FlatHIMRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "atdog_dog_w1_wall_him"
        self.max_iterations = 10000
