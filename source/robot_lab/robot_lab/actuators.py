"""Delayed PD with fixed-shape buffer operations, preserving per-joint delays."""
import torch
from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils import DelayBuffer


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
