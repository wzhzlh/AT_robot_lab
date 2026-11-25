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
