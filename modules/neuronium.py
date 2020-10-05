import random


class Neuronium:

  def __init__(self, quantity): 
    self.w = [random.uniform(0, 1) for i in range(quantity)]
    self.old_w = self.w[:] # to momentum
    self.inputs = None # x => inputs
    self.output = None # I => output no activation

  def __repr__(self): 
    return " <-> ".join([str(elem) for elem in self.w])
    # return " <-> ".join(str(self.output))

  def __str__(self):
    return " <-> ".join([str(elem) for elem in self.w])
    # return " <-> ".join(str(self.output))
