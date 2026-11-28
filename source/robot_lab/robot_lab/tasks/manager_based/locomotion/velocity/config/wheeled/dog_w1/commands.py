"""DogW1 command semantics: stopping means zero translation AND zero yaw."""

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.commands import UniformThresholdVelocityCommand


class DogW1VelocityCommand(UniformThresholdVelocityCommand):
    def _update_command(self):
        # Preserve heading and pure-yaw commands. The upstream is_standing_env
        # mask, rather than translation magnitude, defines sampled full stops.
        UniformVelocityCommand._update_command(self)
