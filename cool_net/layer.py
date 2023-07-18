from cool_net import Neuron
from cool_net import Funcs

class Layer:
  activation_funcs = ["linear","relu","sigmoid", "tanh","leaky_relu"]

  def __init__(self, 
               input_dimension, 
               neuron_count, 
               activation_func,
               weight_init_technique="random",
               bias_init_technique="zero"):
    if activation_func not in Layer.activation_funcs:
      raise Exception(f"Error: activation function {activation_func} is not valid.")
    self.activation_func = activation_func
    if weight_init_technique not in Neuron.techniques_weight_init:
      print(f"Warning: weight initialization technique {weight_init_technique} not valid. using {Neuron.default_weight_init}")
      weight_init_technique = Neuron.default_weight_init
    if bias_init_technique not in Neuron.techniques_bias_init:
      print(f"Warning: bias initialization technqiue {bias_init_technique} is not valid. using {Neuron.default_bias_init}.")
      bias_init_technique = Neuron.default_bias_init
    self.n_count = neuron_count
    self.n = [Neuron(input_dimension,weight_init_technique,bias_init_technique) for i in range(self.n_count)]

  def forward(self, x):
    return [neuron.forward(x) for neuron in self.n]
  
  def forward_pass(self, x):
    outputs = self.forward(x)
    if self.activation_func == "linear":
      return [Funcs.linear(output) for output in outputs]
    if self.activation_func == "relu":
      return [Funcs.relu(output) for output in outputs]
    if self.activation_func == "sigmoid":
      return [Funcs.sigmoid(output) for output in outputs]
    if self.activation_func == "tanh":
      return [Funcs.tanh(output) for output in outputs]
    if self.activation_func == "leaky_relu":
      return [Funcs.leaky_relu(output) for output in outputs]
    
  def load_params(self, params: dict):
    for i in range(self.n_count):
      self.n[i].load_weights(params[i]['w'])
      self.n[i].load_bias(params[i]['b'])
  
  def get_params(self) -> dict:
    params = {}
    for i in range(self.n_count):
      params[i] = {}
      params[i]["b"] = self.n[i].get_bias()
      params[i]["w"] = self.n[i].get_weights()
    return params
  