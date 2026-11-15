"""Compare real upstream delayed PD and buffer methods on CPU and CUDA."""
import ast
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import isaaclab

vendor = Path(isaaclab.__path__[0])
ns = dict(torch=torch, Sequence=Sequence, ArticulationActions=NS)


def load(path, names):
    tree = ast.parse(path.read_text())
    nodes = ast.parse('from __future__ import annotations').body
    nodes += [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)


load(vendor / 'utils/buffers/circular_buffer.py', ['CircularBuffer'])
load(vendor / 'utils/buffers/delay_buffer.py', ['DelayBuffer'])
class Base:
    def reset(self, ids):
        pass
    def _clip_effort(self, effort):
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
ns['ActuatorBase'] = Base
load(vendor / 'actuators/actuator_pd.py', ['IdealPDActuator', 'DelayedPDActuator'])
load(Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/actuators.py',
     ['StreamDelayBuffer', 'StreamDelayedPDActuator'])

for device in ('cpu', 'cuda:0'):
    for delay in (0, 1, 2, 5):
        old = ns['DelayBuffer'](delay, 64, device)
        new = ns['StreamDelayBuffer'](delay, 64, device)
        for step in range(40):
            if step in (0, 4, 19, 32):
                ids = None if step in (0, 32) else torch.tensor([1, 7, 20], device=device)
                count = 64 if ids is None else len(ids)
                lag = torch.randint(0, delay + 1, (count,), device=device, dtype=torch.int)
                for buffer in (old, new):
                    buffer.reset(ids)
                    buffer.set_time_lag(lag, ids)
            data = torch.randn(64, 3, device=device)
            expected, actual = old.compute(data), new.compute(data)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(new._circular_buffer._buffer, old._circular_buffer._buffer, rtol=0, atol=0)
            actual.zero_()  # Outputs must not alias history.
            torch.testing.assert_close(new._circular_buffer._buffer, old._circular_buffer._buffer, rtol=0, atol=0)
    # Use real upstream reset/compute with 12 independent groups and sampled
    # delays; equal seeds must retain the same reset RNG consumption.
    pairs = []
    for joint in range(12):
        pair = []
        for cls, buffer_cls in ((ns['DelayedPDActuator'], ns['DelayBuffer']),
                                (ns['StreamDelayedPDActuator'], ns['StreamDelayBuffer'])):
            a = object.__new__(cls)
            a.cfg = NS(min_delay=1, max_delay=2)
            a._num_envs, a._device = 64, device
            a.stiffness = torch.full((64, 1), 25., device=device)
            a.damping = torch.full((64, 1), .5, device=device)
            a.effort_limit = torch.full((64, 1), 23.5, device=device)
            for kind in ('positions', 'velocities', 'efforts'):
                setattr(a, kind + '_delay_buffer', buffer_cls(2, 64, device))
            pair.append(a)
        pairs.append(pair)
    for step in range(32):
        if step in (0, 7, 21):
            ids = None if step == 0 else torch.tensor([0, 4, 17], device=device)
            for joint, pair in enumerate(pairs):
                for actuator in pair:
                    torch.manual_seed(1000 + step * 12 + joint)
                    actuator.reset(ids)
        for old, new in pairs:
            values = [torch.randn(64, 1, device=device) * 3 for _ in range(5)]
            actions = [NS(joint_positions=values[0].clone(), joint_velocities=values[1].clone(),
                          joint_efforts=values[2].clone()) for _ in range(2)]
            old.compute(actions[0], values[3], values[4])
            new.compute(actions[1], values[3], values[4])
            torch.testing.assert_close(new.computed_effort, old.computed_effort, rtol=0, atol=0)
            torch.testing.assert_close(new.applied_effort, old.applied_effort, rtol=0, atol=0)
            torch.testing.assert_close(new.positions_delay_buffer.time_lags, old.positions_delay_buffer.time_lags)
    print(device, 'buffer contents, resets, independent sampled delays, PD torques exactly match', flush=True)
