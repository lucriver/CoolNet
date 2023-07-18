import unittest
import numpy as np
from cool_net import Layer

class LayerTests(unittest.TestCase):
  def test_get_params_type(self):
    layer = Layer(3,4,'relu')
    self.assertIsInstance(layer.get_params(),dict)

  def test_load_params(self):
    input_dimension = 2
    neuron_count = 3
    expected_bias = 0.01
    expected_params = {}
    for neuron_index in range(neuron_count):
      expected_params[neuron_index] = {}
      expected_params[neuron_index]['w'] = np.random.randn(input_dimension)
      expected_params[neuron_index]['b'] = expected_bias
    layer = Layer(input_dimension,neuron_count,'relu')
    layer.load_params(expected_params)
    self.assertDictEqual(expected_params,layer.get_params())


    
if __name__ == "__main__":
  unittest.main()