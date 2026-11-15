"""Numerical CUDA regression for Warp/Torch stream interoperation, no simulation."""
import ast
import importlib.util
import sys
from pathlib import Path
import torch
import warp as wp
# Load the real installed Warp kernel and wrapper without importing USD/App.
import isaaclab
vendor = Path(isaaclab.__path__[0]) / 'utils/warp'
spec = importlib.util.spec_from_file_location('vendor_raycast_kernels', vendor / 'kernels.py')
kernels = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernels
spec.loader.exec_module(kernels)
ops = ast.parse((vendor / 'ops.py').read_text())
ns = dict(torch=torch, wp=wp, kernels=kernels)
exec(compile(ast.Module(body=[n for n in ops.body if isinstance(n, ast.FunctionDef) and n.name == 'raycast_mesh'], type_ignores=[]), str(vendor / 'ops.py'), 'exec'), ns)
reference = ns['raycast_mesh']

path = Path(__file__).resolve().parents[1] / 'source/robot_lab/robot_lab/terrain_raycast.py'
tree = ast.parse(path.read_text())
ns = dict(torch=torch, wp=wp, kernels=kernels, _synchronous_raycast=reference)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(path), 'exec'), ns)
from types import SimpleNamespace
module = SimpleNamespace(raycast_mesh=ns['raycast_mesh'])
wp.init()
points = wp.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]], dtype=wp.vec3, device='cuda:0')
indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device='cuda:0')
mesh = wp.Mesh(points=points, indices=indices)
for use_other_stream in (False, True):
    stream = torch.cuda.Stream() if use_other_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(8):
            starts = torch.rand(8, 4, 18, 3, device='cuda') * 4 - 2
            starts[..., 2] = 1.
            directions = torch.zeros_like(starts)
            directions[..., 2] = -1
            # Upstream wrapper launches Warp's default stream. Establish its
            # dependencies explicitly when comparing a non-default Torch stream.
            stream.synchronize()
            with torch.cuda.stream(torch.cuda.default_stream()):
                expected = reference(starts, directions, mesh, max_dist=2., return_normal=True)
            # The oracle allocates on the default stream, but comparisons use
            # the test stream. Protect its storage from cross-stream reuse.
            for value in expected:
                if value is not None:
                    value.record_stream(stream)
            actual = module.raycast_mesh(starts, directions, mesh, max_dist=2., return_normal=True)
            # Consume immediately on the same stream, including temporary
            # tensor allocation churn, to catch missing stream dependencies.
            consumed = actual[0] + torch.zeros_like(actual[0])
            torch.testing.assert_close(consumed, expected[0])
            torch.testing.assert_close(actual[2], expected[2])
            assert actual[1] is None and actual[3] is None
    stream.synchronize()
print('CUDA raycast: 16 randomized batches passed on default and non-default streams')

# Compare the actual sensor update methods, including partial updates, misses,
# and per-environment vertical drift. No simulator lifecycle is needed here.
sensor_path = Path(isaaclab.__path__[0]) / 'sensors/ray_caster/ray_caster.py'
sensor_tree = ast.parse(sensor_path.read_text())
vendor_class = next(n for n in sensor_tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayCaster')
vendor_update = next(n for n in vendor_class.body if isinstance(n, ast.FunctionDef) and n.name == '_update_buffers_impl')
vendor_update.returns = None
for arg in vendor_update.args.args:
    arg.annotation = None
ray_class = SimpleNamespace(meshes={'terrain': mesh})
sensor_ns = dict(raycast_mesh=reference, RayCaster=ray_class)
exec(compile(ast.Module(body=[vendor_update], type_ignores=[]), str(sensor_path), 'exec'), sensor_ns)
sensor_path = path.with_name('terrain_sensor.py')
sensor_tree = ast.parse(sensor_path.read_text())
local_ns = dict(raycast_mesh=module.raycast_mesh, RayCaster=type('RayCaster', (), {'meshes': ray_class.meshes}))
exec(compile(ast.Module(body=[n for n in sensor_tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]), str(sensor_path), 'exec'), local_ns)
for ids in (slice(None), torch.tensor([0, 3, 7], device='cuda')):
    starts = torch.rand(8, 9, 3, device='cuda') * 4 - 2
    starts[..., 2] = 1.
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    drift = torch.rand(8, 3, device='cuda') * .02
    def sensor():
        return SimpleNamespace(
            _ray_starts_w=starts, _ray_directions_w=directions, ray_cast_drift=drift,
            _data=SimpleNamespace(ray_hits_w=torch.full_like(starts, -123.)),
            cfg=SimpleNamespace(max_distance=2., mesh_prim_paths=['terrain']),
            _update_ray_infos=lambda env_ids: None,
        )
    old, new = sensor(), sensor()
    torch.cuda.synchronize()
    sensor_ns['_update_buffers_impl'](old, ids)
    local_ns['StreamRayCaster']._update_buffers_impl(new, ids)
    torch.testing.assert_close(new._data.ray_hits_w, old._data.ray_hits_w, rtol=0, atol=0)
print('Sensor update: full/subset updates, drift, misses, and untouched rows match upstream exactly')
