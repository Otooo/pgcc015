from modules import Neuronium


class Layer:

  def __init__(self, previousL=None, nextL=None): 
    self.neuroniums = []
    self.previousL = previousL
    self.nextL = nextL
    self.label = ''

  def __repr__(self): 
    toply = ''
    # toply += " <-> ".join([str(elem.output) for elem in self.neuroniums]) + '\n'
    for n in self.neuroniums:
      toply += '\n' + " <-> ".join([str(elem) for elem in n.w])
    
    return toply

  def __str__(self):
    toply = ''
    # toply += " <-> ".join([str(elem.output) for elem in self.neuroniums]) + '\n'
    for n in self.neuroniums:
      toply += '\n' + " <-> ".join([str(elem) for elem in n.w])
    
    return toply
