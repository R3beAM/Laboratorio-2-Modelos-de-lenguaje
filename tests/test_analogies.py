import unittest

import numpy as np

from analogies import solve_analogy


class SolveAnalogyTests(unittest.TestCase):
    def setUp(self):
        # Construct a toy embedding space where analogies are easy to verify.
        self.embeddings = {
            "man": np.array([1.0, 0.0, 0.0]),
            "woman": np.array([1.0, 1.0, 0.0]),
            "king": np.array([2.0, 0.0, 1.0]),
            "queen": np.array([2.0, 1.0, 1.0]),
            "prince": np.array([2.0, -1.0, 1.0]),
        }

    def test_basic_analogy(self):
        result = solve_analogy("man", "king", "woman", self.embeddings)
        self.assertEqual(result, "queen")

    def test_top_k_results(self):
        results = solve_analogy("man", "king", "woman", self.embeddings, top_k=2)
        self.assertEqual(results[0][0], "queen")
        self.assertEqual(len(results), 2)

    def test_missing_word(self):
        with self.assertRaises(ValueError):
            solve_analogy("unknown", "king", "woman", self.embeddings)

    def test_inconsistent_dimensions(self):
        embeddings = dict(self.embeddings)
        embeddings["rogue"] = np.array([[0.0]])
        with self.assertRaises(ValueError):
            solve_analogy("man", "king", "woman", embeddings)

    def test_zero_length_target_vector(self):
        embeddings = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([1.0, 0.0]),
            "c": np.array([1.0, 0.0]),
            "d": np.array([0.0, 1.0]),
        }
        with self.assertRaises(ValueError):
            solve_analogy("a", "b", "c", embeddings)


if __name__ == "__main__":
    unittest.main()
