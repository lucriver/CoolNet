import numpy as np

class Funcs():
  def __init__(self):
    pass

  @staticmethod
  def linear(x):
    return x
  
  @staticmethod
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
  @staticmethod
  def relu(x):
    return np.maximum(0, x)
  
  @staticmethod
  def leaky_relu(x, alpha=0.01):
      return np.where(x > 0, x, alpha*x)

  @staticmethod
  def tanh(x):
    return np.tanh(x)

  @staticmethod
  class Deriv:
    def __init__(self):
      pass
      
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def tanh(x):
      return (1 - x) * (1 + x)
    
    @staticmethod
    def sigmoid(sigmoid_x):
      return sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def relu(x):
      if x > 0:
          return 1
      else:
          return 0
      
    @staticmethod
    def linear(x):
      return x
      
  class Loss:
    def __init__(self):
      pass

    @staticmethod
    def binaryCrossEntropy(y_true, y_pred):
      epsilon = 1e-15  # small constant to avoid log(0)
      y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predicted probabilities to avoid log(0) or log(1)
      loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
      return np.mean(loss)

    @staticmethod
    def categoricalCrossEntropy(y_true, y_pred):
      return -np.sum(y_true * np.log(y_pred))

