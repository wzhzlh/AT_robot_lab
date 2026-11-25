"""Delayed PD with fixed-shape buffer operations, preserving per-joint delays."""
import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer, configclass


class StreamDelayBuffer(DelayBuffer):
    """Equivalent append/read without tensor-valued Python conditions."""

    def compute(self, data):
        buffer = self._circular_buffer
        if data.shape[0] != self.batch_size:
            raise ValueError('Delay buffer batch size mismatch')
        data = data.to(self.device)
        capacity = self._history_length + 1
        if buffer._buffer is None:
            buffer._pointer = -1
            buffer._buffer = torch.empty((capacity, *data.shape), dtype=data.dtype, device=self.device)
        buffer._pointer = (buffer._pointer + 1) % capacity
        buffer._buffer[buffer._pointer].copy_(data)
        first = buffer._num_pushes == 0
        mask = first.reshape(1, self.batch_size, *([1] * (data.ndim - 1)))
        torch.where(mask, data.unsqueeze(0), buffer._buffer, out=buffer._buffer)
        buffer._num_pushes += 1
        # Every row has just been appended, so the upstream empty check is
        # unnecessary. Indexing allocates an independent output already.
        lag = torch.minimum(self._time_lags, buffer._num_pushes - 1)
        index = torch.remainder(buffer._pointer - lag, capacity)
        return buffer._buffer[index, buffer._ALL_INDICES]


class StreamDelayedPDActuator(DelayedPDActuator):
    """Keep upstream reset sampling, gains, saturation, and PD computation."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = StreamDelayBuffer(cfg.max_delay, self._num_envs, self._device)
        self.velocities_delay_buffer = StreamDelayBuffer(cfg.max_delay, self._num_envs, self._device)
        self.efforts_delay_buffer = StreamDelayBuffer(cfg.max_delay, self._num_envs, self._device)


def directional_friction_torque(velocity, a_positive, b_positive, a_negative, b_negative):
    """Resisting joint torque in Nm; velocity in rad/s, zero torque at rest.

    The measured moving-friction fit is |tau| = a * |velocity| + b.
    Direction refers to the simulated joint coordinate, not robot travel direction.
    No static/breakaway friction was identified by this fit.
    """
    slope = torch.where(velocity >= 0.0, a_positive, a_negative)
    intercept = torch.where(velocity >= 0.0, b_positive, b_negative)
    return -torch.sign(velocity) * (slope * velocity.abs() + intercept)


def _measured_friction(cfg, velocity):
    return directional_friction_torque(
        velocity, cfg.friction_a_positive, cfg.friction_b_positive,
        cfg.friction_a_negative, cfg.friction_b_negative,
    )


class FrictionDelayedPDActuator(StreamDelayedPDActuator):
    """Delayed leg controller plus measured passive friction after motor saturation."""

    def compute(self, control_action, joint_pos, joint_vel):
        action = super().compute(control_action, joint_pos, joint_vel)
        # Keep motor effort diagnostics separate from the passive resistance.
        action.joint_efforts = action.joint_efforts + _measured_friction(self.cfg, joint_vel)
        return action


class FrictionImplicitActuator(ImplicitActuator):
    """Keep the PhysX wheel velocity drive; add passive friction as external effort."""

    def compute(self, control_action, joint_pos, joint_vel):
        # Implicit PD remains in PhysX. Only passive friction is added here.
        control_action.joint_efforts = control_action.joint_efforts + _measured_friction(self.cfg, joint_vel)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class FrictionDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type = FrictionDelayedPDActuator
    friction_a_positive: float = 0.0107065786
    friction_b_positive: float = 0.231393197
    friction_a_negative: float = 0.0122314048
    friction_b_negative: float = 0.21220256


@configclass
class FrictionImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type = FrictionImplicitActuator
    friction_a_positive: float = 0.0107065786
    friction_b_positive: float = 0.231393197
    friction_a_negative: float = 0.0122314048
    friction_b_negative: float = 0.21220256
