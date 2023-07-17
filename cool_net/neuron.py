import numpy as np

class Neuron:
  def __init__(self, in_dim, bias=None):
    self.weight_count = in_dim
    self.w = np.random.randn(self.weight_count) * .01
    self.b = bias
    if self.b == None:
      self.b = np.random.randn() * 0.01

  def forward(self, x):
    return np.dot(x,self.w) + self.b

  def zeroWeights(self):
    self.w = np.zeros(len(self.weight_count))

  def zeroBias(self):
    self.b = 0

  def loadWeights(self, weightsDict):
    if len(weightsDict) != self.weight_count:
      raise Exception("Error: Mismatch in weights to load versus the quantity of weights for the neuron.")
    for i in range(self.w_count):
      self.w[i] = weightsDict[i]

  def loadBias(self, bias_val):
    self.b = bias_val

  def getBias(self):
    return self.b
  
  def getWeightsDict(self):
    return {index: weight for index, weight in enumerate(self.w)}
  
  def getWeightsList(self):
    return self.w