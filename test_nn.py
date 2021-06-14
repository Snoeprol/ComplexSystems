import unittest
from rabbit import get_nn
import numpy as np

class TestNearestNeighbor(unittest.TestCase):

    def test_stack(self):
        N = 100
        # Stack of rabbits at zero
        rabbits = np.zeros((N, 4))
        nn_array = np.zeros((N, N, 2), dtype = np.int)
        com_array = np.zeros((N, N), dtype = np.int)
        nn_indices = np.zeros((N), dtype = np.int)
        nn_array = get_nn(nn_array[:, :, 0], com_array, nn_indices, rabbits, 0.1)
        
        # Self is excluded hence N * N - N
        self.assertEqual(np.count_nonzero(nn_array), int(N * N - N), "Should be 0")
        
if __name__ == '__main__':
    unittest.main()