import numpy as np

class Neuron:
  techniques_weight_init = ["zero","random"]
  default_weight_init = "random"
  techniques_bias_init = ["zero","small","one"]
  default_bias_init = "zero"

  def __init__(self, 
               input_dim, 
               weight_init_technique="random",
               bias_init_technique="zero"):
    self.weight_init_technique = weight_init_technique
    if weight_init_technique not in Neuron.techniques_weight_init:
      print("weight initialization method not supported. using",Neuron.default_weight_init)
      self.weight_init_technique = Neuron.default_bias_init
    self.bias_init_technique = bias_init_technique
    if bias_init_technique not in Neuron.techniques_bias_init:
      print("bias initialization method not supported. using",Neuron.default_bias_init)
      self.bias_init_technique = Neuron.default_bias_init

    self.w_count = input_dim
    self.w = self._init_weights(self.w_count, self.weight_init_technique)
    self.b = self._init_bias(self.bias_init_technique)

  def _init_weights(self, weight_count, init_technique):
    if init_technique == "zero":
      return np.zeros(weight_count)
    if init_technique == "random":
      return np.random.randn(weight_count) * .01
    
  def _init_bias(self, init_technique):
    if init_technique == "zero":
      return 0
    if init_technique == "small":
      return 0.01
    if init_technique == "one":
      return 1

  def forward(self, x):
    return np.dot(x,self.w) + self.b

  def load_weights(self, weights: np.array):
    if len(weights) != self.w_count:
      raise Exception("Error: Mismatch in weights to load versus the quantity of weights for the neuron.")
    self.w = weights

  def load_bias(self, bias_val):
    self.b = bias_val

  def get_weights(self):
    return self.w

  def get_bias(self):
    return self.b
  
  def get_params(self) -> dict:
    return {'w': self.getWeights(), 'b': self.getBias()}