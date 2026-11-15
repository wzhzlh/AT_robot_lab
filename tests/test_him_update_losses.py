"""CPU regression for deferred loss logging, without simulator imports."""

import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

HIM = Path(__file__).resolve().parents[1] / "source/robot_lab/robot_lab/him"
PACKAGE = "him_cpu_loss_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(HIM)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", HIM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Estimator = load("him_estimator").HIMEstimator
Actor = load("him_actor_critic").HIMActorCritic
PPO = load("him_ppo").HIMPPO


class LossLoggingTests(unittest.TestCase):
    def test_estimator_tensor_and_float_paths_produce_same_update(self):
        torch.manual_seed(4)
        estimator = Estimator(temporal_steps=1, num_one_step_obs=5, vel_slice=(0, 3),
                              target_slices=((3, 8),), enc_hidden_dims=(8, 4), tar_hidden_dims=(8,),
                              num_prototypes=4)
        tensor_estimator = copy.deepcopy(estimator)
        obs, target = torch.randn(8, 5), torch.randn(8, 8)
        mask = torch.tensor([True, False, True, True, False, True, True, True])
        expected = estimator.update(obs, target, valid_mask=mask)
        actual = tensor_estimator.update(obs, target, valid_mask=mask, return_tensors=True)
        self.assertTrue(all(not value.requires_grad for value in actual))
        for left, right in zip(expected, actual):
            self.assertEqual(left, right.item())
        for left, right in zip(estimator.parameters(), tensor_estimator.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        before = copy.deepcopy(tensor_estimator.state_dict())
        zeros = tensor_estimator.update(obs, target, valid_mask=torch.zeros(8, dtype=torch.bool), return_tensors=True)
        self.assertEqual([value.item() for value in zeros], [0.0, 0.0])
        for key, value in before.items():
            torch.testing.assert_close(tensor_estimator.state_dict()[key], value, rtol=0, atol=0)

    def test_ppo_returns_float_losses_matching_estimator_batch_means(self):
        torch.manual_seed(8)
        actor = Actor(num_actor_obs=5, num_one_step_obs=5, num_critic_obs=8, num_actions=2,
                      actor_hidden_dims=(8,), critic_hidden_dims=(8,),
                      estimator_encoder_hidden_dims=(8, 4), estimator_target_hidden_dims=(8,),
                      estimator_num_prototypes=4, estimator_vel_slice=(0, 3),
                      estimator_target_slices=((3, 8),))
        ppo = PPO(actor, num_learning_epochs=2, num_mini_batches=2)
        ppo.init_storage(4, 2, [5], [8], [2])
        with torch.inference_mode():
            for _ in range(2):
                ppo.act(torch.randn(4, 5), torch.randn(4, 8))
                ppo.process_env_step(torch.ones(4), torch.zeros(4), {}, torch.randn(4, 8),
                                     torch.tensor([True, True, False, True]))
            ppo.compute_returns(torch.randn(4, 8))
        observed = []
        original = actor.estimator.update

        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            observed.append(torch.stack(result))
            return result

        actor.estimator.update = record
        losses = ppo.update()
        self.assertEqual(len(losses), 5)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))
        self.assertTrue(torch.isfinite(torch.tensor(losses)).all())
        torch.testing.assert_close(torch.tensor(losses[2:4]), torch.stack(observed).mean(0))
        self.assertEqual(ppo.storage.step, 0)


if __name__ == "__main__":
    unittest.main()
