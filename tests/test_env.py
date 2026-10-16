"""Run with: python -m unittest discover -s tests"""
import unittest
import numpy as np

from neurosim.env import RocketLandingEnv


class RocketEnvironmentTests(unittest.TestCase):
    def test_all_presets_reset_and_step(self):
        for planet in ("earth", "mars", "moon"):
            for vehicle in ("falcon9_inspired", "custom"):
                env = RocketLandingEnv(planet=planet, vehicle=vehicle)
                obs, _ = env.reset(seed=42)
                self.assertEqual(obs.shape, (8,))
                self.assertTrue(env.observation_space.contains(obs))
                obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0]))
                self.assertTrue(np.isfinite(reward))
                self.assertTrue(env.observation_space.contains(obs))
                self.assertIn("outcome", info)
                env.close()

    def test_rcs_changes_rotation(self):
        a = RocketLandingEnv()
        b = RocketLandingEnv()
        a.reset(seed=99)
        b.reset(seed=99)
        a.step(np.array([0.0, 0.0, 1.0]))
        b.step(np.array([0.0, 0.0, -1.0]))
        self.assertGreater(a.state[5], b.state[5])
        self.assertEqual(a.action_space.shape, (3,))
        a.close()
        b.close()

    def test_seed_reproducibility(self):
        env = RocketLandingEnv()
        a, _ = env.reset(seed=123)
        b, _ = env.reset(seed=123)
        np.testing.assert_array_equal(a, b)
        env.close()


if __name__ == "__main__":
    unittest.main()
