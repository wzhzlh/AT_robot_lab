"""Wheel angle masking for dog_w1 observations in explicit joint order."""


def joint_pos_rel_without_wheel(env, asset_cfg, wheel_asset_cfg):
    """Mask wheel angles after selecting joints in action order.

    Resolved wheel IDs refer to the articulation; columns in the selected observation
    may have a different order, so translate the IDs before masking.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    wheel_columns = [asset_cfg.joint_ids.index(joint_id) for joint_id in wheel_asset_cfg.joint_ids]
    joint_pos[:, wheel_columns] = 0.0
    return joint_pos


def dog_w1_joint_pos_scale():
    """Mirror hip abduction across left/right using the URDF joint axes."""
    return {
        "FL_hip_joint": 0.125,
        "FR_hip_joint": -0.125,
        "RL_hip_joint": -0.125,
        "RR_hip_joint": 0.125,
        "FL_thigh_joint": 0.25,
        "FR_thigh_joint": 0.25,
        "RL_thigh_joint": 0.25,
        "RR_thigh_joint": 0.25,
        "FL_calf_joint": 0.25,
        "FR_calf_joint": 0.25,
        "RL_calf_joint": 0.25,
        "RR_calf_joint": 0.25,
    }


def dog_w1_joint_vel_scale():
    """Make a positive wheel action drive every wheel toward body +x."""
    return {
        "FL_foot_joint": 5.0,
        "FR_foot_joint": -5.0,
        "RL_foot_joint": 5.0,
        "RR_foot_joint": -5.0,
    }
