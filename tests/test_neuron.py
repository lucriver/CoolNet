import unittest
import numpy as np
from cool_net.neuron import Neuron

class NeuronTests(unittest.TestCase):
  def test_load_weights(self):
    input_dimension = 3
    expected_weights = np.random.randn(input_dimension)
    n = Neuron(input_dimension)
    n.load_weights(expected_weights)
    self.assertEqual(n.get_weights().all(),expected_weights.all())
    
if __name__ == "__main__":
  unittest.main()