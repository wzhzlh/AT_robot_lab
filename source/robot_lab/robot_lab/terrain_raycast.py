"""Terrain rays on the PyTorch CUDA stream, without a device-wide wait."""
import torch
import warp as wp
from isaaclab.utils.warp import kernels
from isaaclab.utils.warp import raycast_mesh as _synchronous_raycast


def raycast_mesh(ray_starts, ray_directions, mesh, max_dist=1e6,
                 return_distance=False, return_normal=False, return_face_id=False):
    """Match Isaac Lab's ray result contract; launch on the consumer's stream.

    Restrict the fast path to same-device CUDA tensors. CPU/cross-device calls
    retain the upstream implementation. No global synchronization is removed
    without establishing stream ordering between Torch and Warp.
    """
    device = ray_starts.device
    if (device.type != "cuda" or wp.device_to_torch(mesh.device) != device
            or ray_directions.device != device):
        return _synchronous_raycast(ray_starts, ray_directions, mesh, max_dist=max_dist,
                                   return_distance=return_distance, return_normal=return_normal,
                                   return_face_id=return_face_id)
    shape = ray_starts.shape
    starts = ray_starts.reshape(-1, 3).contiguous()
    directions = ray_directions.reshape(-1, 3).contiguous()
    count = starts.shape[0]
    hits = torch.full_like(starts, float('inf'))
    distance = torch.full((count if return_distance else 1,), float('inf'), device=device)
    normal = torch.full((count if return_normal else 1, 3), float('inf'), device=device)
    face = torch.full((count if return_face_id else 1,), -1, dtype=torch.int32, device=device)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    wp.launch(
        kernels.raycast_mesh_kernel, dim=count,
        inputs=[mesh.id, wp.from_torch(starts, dtype=wp.vec3), wp.from_torch(directions, dtype=wp.vec3),
                wp.from_torch(hits, dtype=wp.vec3), wp.from_torch(distance, dtype=wp.float32),
                wp.from_torch(normal, dtype=wp.vec3), wp.from_torch(face, dtype=wp.int32),
                float(max_dist), int(return_distance), int(return_normal), int(return_face_id)],
        device=mesh.device, stream=stream,
    )
    return (hits.view(shape), distance.view(shape[:-1]) if return_distance else None,
            normal.view(shape) if return_normal else None, face.view(shape[:-1]) if return_face_id else None)
