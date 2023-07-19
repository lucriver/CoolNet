from cool_net.layer import Layer
from cool_net.funcs import Funcs
import numpy as np 

class Net:

  def __init__(self, 
              input_dimension, 
              hidden_layer_count,
              hidden_neuron_count,
              hidden_activ_func, 
              output_neuron_count,
              output_activ_func):
    self.h_activ_func = hidden_activ_func
    self.o_activ_func = output_activ_func
    if input_dimension <= 0:
      raise Exception("Error: Invalid number of input features.")
    if hidden_layer_count < 0:
      raise Exception("Error: Hidden layers must be greater than or equal to 0")
    if hidden_layer_count == 0:
      self.layer = [Layer(input_dimension,
                          output_neuron_count,
                          self.o_activ_func)]
    elif hidden_layer_count == 1:
      self.layer = [Layer(input_dimension,hidden_neuron_count,self.h_activ_func),
                    Layer(hidden_neuron_count,output_neuron_count,self.o_activ_func)]
    else:
      input_layer = [Layer(input_dimension,hidden_neuron_count,self.h_activ_func)]
      hidden_layers = [Layer(hidden_neuron_count,hidden_neuron_count,self.h_activ_func) for i in range(hidden_layer_count-1)]
      output_layer = [Layer(hidden_neuron_count,output_neuron_count,self.o_activ_func)]
      self.layer = input_layer + hidden_layers + output_layer
    self.layer_count = len(self.layer)

  def forward_with_layers(self,x):
    outputs = []
    for i in range(self.layer_count):
      y_before_activ_func = self.layer[i].forward(x)
      outputs.append(y_before_activ_func)
      x = self.layer[i].forward_pass(x)
    return outputs
 
  def forward_pass_with_layers(self,x):
    outputs = []
    for i in range(self.layer_count):
      x = self.layer[i].forward_pass(x)
      outputs.append(x)
    return outputs

  def forward_pass(self,x):
    for layer in self.layer:
      x = layer.forward_pass(x)
    return x
  
  def load_params(self, net_params):
    if len(net_params) != self.layer_count:
      raise Exception("Error: mismatch in network layers against loaded network parameter layers.")
    for layer_index in range(self.layer_count):
      if len(net_params[layer_index]) != self.layer[layer_index].n_count:
        raise Exception("Error: mismatch in quantity of layer neurons to loaded layer neurons.")
      self.layer[layer_index].load_params(net_params[layer_index])

  def get_params(self):
    net_params = {}
    for layer_index in range(self.layer_count):
      net_params[layer_index] = self.layer[layer_index].get_params()
    return net_params
  

  def fit(self,
          train_data,
          test_data,
          epochs,
          learning_rate,
          loss_metric,
          loss_metric_freq
          ):
    sample_index_input = 0
    sample_index_groundtruth = 1
    train_data_count = len(train_data)
    test_data_count = len(test_data)

    for epoch in range(epochs):

      if epoch % loss_metric_freq == 0:
        losses = []
        loss = None
        for i in range(test_data_count):
          output = self.forward_pass(test_data[i][sample_index_input])
          if loss_metric == "BCE":
            loss = Funcs.Loss.binaryCrossEntropy(test_data[i][sample_index_groundtruth],output)
          losses.append(loss)
        print(f"epoch {epoch} loss:",sum(losses)/test_data_count)

      for i in range(train_data_count):
        # compute forward passes
        l_outs_noActivFunc = self.forward_with_layers(train_data[i][sample_index_input])
        l_outs = self.forward_pass_with_layers(train_data[i][sample_index_input])

        # compute output layer gradients
        out_gradient = []
        for index_layer_neuron in range(self.layer[-1].n_count):
          operand = train_data[i][sample_index_groundtruth] - l_outs[-1][index_layer_neuron]

          # compute gradient for a neuron in output layer
          if self.o_activ_func == "sigmoid":
            out_gradient.append(Funcs.Deriv.sigmoid(l_outs[-1][index_layer_neuron]) * operand)
          if self.o_activ_func == "relu":
            out_gradient.append(Funcs.Deriv.relu(l_outs[-1][index_layer_neuron]) * operand)
          if self.o_activ_func == "leaky_relu":
            out_gradient.append(Funcs.Deriv.leaky_relu(l_outs[-1][index_layer_neuron]) * operand)
          if self.o_activ_func == "linear":
            out_gradient.append(Funcs.Deriv.linear(l_outs[-1][index_layer_neuron]) * operand)
          if self.o_activ_func == "tanh":
            out_gradient.append(Funcs.Deriv.tanh(l_outs[-1][index_layer_neuron]) * operand)
                    
        # compute gradients for hidden layers
        prev_gradient = out_gradient
        hidden_gradients = []
        layer_count = self.layer_count
        for layer_index in range(self.layer_count):
          hidden_gradient = []
          if layer_index == layer_count - 1:
            break
          current_layer = -2 - layer_index
          for neuron_index in range(self.layer[current_layer].n_count):
            res = 0
            for prev_gradient_index in range(len(prev_gradient)):
              res += prev_gradient[prev_gradient_index] * self.layer[current_layer+1].n[prev_gradient_index].w[neuron_index]
            gradient = None
            if self.h_activ_func == "relu":
              gradient = Funcs.Deriv.relu(l_outs_noActivFunc[layer_index][neuron_index]) * res
            if self.h_activ_func == "leaky_relu":
              gradient = Funcs.Deriv.leaky_relu(l_outs_noActivFunc[layer_index][neuron_index]) * res
            if self.h_activ_func == "tanh":
              gradient = Funcs.Deriv.tanh(l_outs_noActivFunc[layer_index][neuron_index]) * res
            if self.h_activ_func == "linear":
              gradient = Funcs.Deriv.linear(l_outs_noActivFunc[layer_index][neuron_index]) * res
            if self.h_activ_func == "sigmoid":
              gradient = Funcs.Deriv.sigmoid(l_outs_noActivFunc[layer_index][neuron_index]) * res
            hidden_gradient.append(gradient)
          hidden_gradients.append(np.array(hidden_gradient))
          prev_gradient = hidden_gradient
        hidden_gradients.append(np.array(out_gradient))

        for index_layer_neuron in range(self.layer[0].n_count):
          self.layer[0].n[index_layer_neuron].w += learning_rate * np.dot(train_data[i][sample_index_input],hidden_gradients[0][index_layer_neuron])
          self.layer[0].n[index_layer_neuron].b += learning_rate * hidden_gradients[0][index_layer_neuron]

        for index_layer in range(1,self.layer_count):
          for index_layer_neuron in range(self.layer[index_layer].n_count):
            self.layer[index_layer].n[index_layer_neuron].w += learning_rate * np.dot(l_outs[index_layer-1],hidden_gradients[index_layer][index_layer_neuron])
            self.layer[index_layer].n[index_layer_neuron].b += learning_rate * hidden_gradients[index_layer][index_layer_neuron]

