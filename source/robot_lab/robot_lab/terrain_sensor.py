"""RayCaster implementation retaining sensor semantics on the Torch stream."""

from isaaclab.sensors import RayCaster

from robot_lab.terrain_raycast import raycast_mesh


class StreamRayCaster(RayCaster):
    """Preserve upstream pose/drift updates; avoid the ray wrapper's global wait."""

    def _update_buffers_impl(self, env_ids):
        self._update_ray_infos(env_ids)
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            max_dist=self.cfg.max_distance,
            mesh=RayCaster.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)
