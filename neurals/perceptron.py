import threading
import logging
from modules import Neuronium
from modules import Layer


class Perceptron:

  def __init__(self, nLearning, quantityLayers=0, maxEpoch=5000): 
    self.nLearning = nLearning
    self.maxEpoch = maxEpoch

    self.layer = Layer()
    if (quantityLayers != 0):
      auxLayer = self.layer
      for i in range(1, quantityLayers):
        nextL = Layer(auxLayer)
        auxLayer.nextL = nextL
        auxLayer = nextL


  def signal(self, outputNeuronium):
    return 1.0 if outputNeuronium >= 0 else -1.0


  def fix(self, w, x, outputSystem, outputSystemTrained):
    return w + self.nLearning * (outputSystem - outputSystemTrained) * x
    

  def perceptron_fit(self, x, outputSystem):
    epoch = 0
    lenW = len(x[0])
    neuronium = Neuronium(lenW) # Create the weights randomly
    # Logging
    print('PESOS INICIAS: ', neuronium.w)

    while True:
      error = False
      for i in range(len(x)):
        u = 0
        for iw in range(lenW):
          u += (neuronium.w[iw] * x[i][iw])

        outputSystemTrained = self.signal(u)
        if (outputSystemTrained != outputSystem[i]):
          for iw in range(lenW):
            neuronium.w[iw] = self.fix(neuronium.w[iw], x[i][iw], outputSystem[i], outputSystemTrained)
          error = True
        
      epoch += 1
      if (error is False or epoch == self.maxEpoch):
        if (epoch == self.maxEpoch):
          print('PARADA LIMITE ÉPOCA')
        break
      
    return neuronium, epoch


  def perceptron_predict(self, x, neuronium):
    outputSystem_predict = []
    lenW = len(neuronium.w)
    for i in range(len(x)):
      predict = 0
      for iw in range(lenW):
        predict += (neuronium.w[iw] * x[i][iw])
        
      y = 'C1' if self.signal(predict) == -1 else 'C2'
      outputSystem_predict.append(y)
    return outputSystem_predict
